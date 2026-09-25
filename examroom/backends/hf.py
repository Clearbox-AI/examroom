"""HuggingFace transformers backend: exact log-likelihoods + generation for local models.

Works for anything `AutoModelForCausalLM` can load, including local checkpoints
(e.g. the 50M–600M ladder) and bitsandbytes-quantised models (`quantization=4bit|8bit`).
"""

from __future__ import annotations

from examroom.backends.base import BackendInfo, Stats, Timer


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
        if quantization in ("4bit", "8bit"):
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=quantization == "4bit", load_in_8bit=quantization == "8bit"
            )
            kwargs["device_map"] = self.device
        else:
            kwargs["dtype"] = torch_dtype

        self.tok = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=trust_remote_code)
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
            n_params=sum(p.numel() for p in self.model.parameters()),
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
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
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
