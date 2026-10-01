"""HuggingFace transformers backend: exact log-likelihoods + generation for local models.

Works for anything `AutoModelForCausalLM` can load, including local checkpoints
(e.g. the 50M–600M ladder) and bitsandbytes-quantised models (`quantization=4bit|8bit`).

Generation wraps the prompt in the tokenizer's chat template when it has one
(`chat_template=auto`, the default; `on` requires one, `off` sends the raw
prompt): instruction-tuned models answer far worse to a bare prompt, and every
chat API applies the template server-side, so this keeps local and API runs
comparable.  `reasoning=off` passes `enable_thinking=False` to the template
(Qwen3-style hybrid-thinking models).  Likelihood scoring never uses the
template: those probes are continuations written for raw text.
"""

from __future__ import annotations

from examroom.backends.base import BackendInfo, Stats, Timer


def generation_text(tok, prompt: str, use_chat: bool, template_kwargs: dict | None = None) -> tuple[str, bool]:
    """(text to tokenize, add_special_tokens) for a generation prompt.

    A chat template already emits BOS and the assistant turn header, so its
    output is tokenized without special tokens (no double BOS)."""
    if not use_chat:
        return prompt, True
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True, **(template_kwargs or {}))
    return text, False


class HFBackend:
    supports_logprobs = True
    supports_generate = True
    parallel_safe = False  # single local model: threads would contend for the GPU/CPU

    def __init__(
        self,
        model: str,
        device: str | None = None,
        dtype: str | None = None,
        quantization: str | None = None,
        revision: str | None = None,
        batch_size: int = 8,
        trust_remote_code: bool = False,
        chat_template: str = "auto",
        reasoning: str | None = None,
        **_,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        torch_dtype = {
            None: torch.float32 if self.device == "cpu" else torch.float16,
            "float32": torch.float32, "fp32": torch.float32,
            "float16": torch.float16, "fp16": torch.float16,
            "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        }[dtype]
        kwargs: dict = {"revision": revision, "trust_remote_code": trust_remote_code}
        if quantization not in (None, "4bit", "8bit"):  # a GGUF quant name would silently run full precision
            raise ValueError(f"hf backend quantization must be 4bit|8bit (bitsandbytes), got {quantization!r}")
        if quantization in ("4bit", "8bit"):
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=quantization == "4bit", load_in_8bit=quantization == "8bit"
            )
            kwargs["device_map"] = self.device
        else:
            kwargs["dtype"] = torch_dtype

        self.tok = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=trust_remote_code)
        has_template = getattr(self.tok, "chat_template", None) is not None
        if chat_template not in ("auto", "on", "off"):
            raise ValueError("chat_template must be auto|on|off")
        if chat_template == "on" and not has_template:
            raise ValueError(f"{model} has no chat template (use chat_template=off for base models)")
        self.use_chat = has_template and chat_template != "off"
        if reasoning not in (None, "off", "on"):
            raise ValueError("reasoning must be off|on for the hf backend")
        self.template_kwargs = {"enable_thinking": reasoning == "on"} if reasoning else {}
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"
        self.model = AutoModelForCausalLM.from_pretrained(model, **kwargs)
        if "quantization_config" not in kwargs:
            self.model.to(self.device)
        self.model.eval()
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

        self.info = BackendInfo(
            backend="hf", model=model, dtype=str(torch_dtype).replace("torch.", ""),
            quantization=quantization, device=self.device, revision=revision,
            # num_parameters() counts packed 4-bit weights at their real size; a raw numel() sum halves them
            n_params=self.model.num_parameters(),
            weights_mb=round(self.model.get_memory_footprint() / 2**20, 1),
            extra={"chat_template": self.use_chat, "reasoning": reasoning},
        )
        self.stats = Stats()

    # -- likelihood ---------------------------------------------------------
    def _encode(self, prompt: str, cont: str) -> tuple[list[int], list[int]]:
        # Encode prompt and continuation separately and concatenate, so the
        # continuation's tokens (and its leading-space convention) are exactly
        # what the author wrote, independent of BPE merges at the boundary.
        p = self.tok(prompt, add_special_tokens=True)["input_ids"]
        c = self.tok(cont, add_special_tokens=False)["input_ids"]
        return p, c

    def loglik(self, prompt, continuations):
        torch = self.torch
        encs = [self._encode(prompt, c) for c in continuations]
        results: list[tuple[float, int]] = [None] * len(encs)  # type: ignore
        with Timer() as t, torch.no_grad():
            for start in range(0, len(encs), self.batch_size):
                chunk = encs[start:start + self.batch_size]
                seqs = [p + c for p, c in chunk]
                L = max(len(s) for s in seqs)
                pad = self.tok.pad_token_id
                ids = torch.tensor([s + [pad] * (L - len(s)) for s in seqs], device=self.device)
                mask = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs], device=self.device)
                logits = self.model(input_ids=ids, attention_mask=mask).logits.float()
                logp = torch.log_softmax(logits[:, :-1], dim=-1)
                targets = ids[:, 1:]
                tok_lp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                for j, (p, c) in enumerate(chunk):
                    s, e = len(p) - 1, len(p) - 1 + len(c)   # positions predicting cont tokens
                    results[start + j] = (float(tok_lp[j, s:e].sum()), len(c))
        self.stats.tick(t.seconds, prompt_tokens=sum(len(p) + len(c) for p, c in encs))
        self._update_mem()
        return results

    # -- generation ---------------------------------------------------------
    def generate(self, prompt, max_tokens, temperature, n):
        torch = self.torch
        text, special = generation_text(self.tok, prompt, self.use_chat, self.template_kwargs)
        enc = self.tok(text, return_tensors="pt", add_special_tokens=special).to(self.device)
        with Timer() as t, torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=max_tokens, do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                num_return_sequences=n, pad_token_id=self.tok.pad_token_id,
            )
        new = out[:, enc["input_ids"].shape[1]:]
        texts = self.tok.batch_decode(new, skip_special_tokens=True)
        self.stats.tick(t.seconds, prompt_tokens=enc["input_ids"].shape[1], generated_tokens=int(new.numel()))
        self._update_mem()
        return texts

    def _update_mem(self):
        if self.device.startswith("cuda"):
            self.stats.peak_device_mem_mb = round(self.torch.cuda.max_memory_allocated() / 2**20, 1)

    def close(self):
        del self.model
        if self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()
