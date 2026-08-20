"""Track 3 — lenient, QUALITATIVE scorer for generated dialogue replies.

Prompt: "A: <opener>\nB:". Scores the continuation on four cheap signals and
returns their mean; use only for trends across the ladder, not as a headline.

  format       the reply is a non-empty line, and any further lines are speaker
               turns ("A:"/"B:") — no prose, no markup
  alternation  speaker turns alternate (B, A, B, …) after the prompt's "B:"
  on_topic     content-word overlap between the reply and the opener, or a
               generic-but-appropriate response word (yes/no/thanks/sure…)
  non_degenerate   not a verbatim copy of the opener, not a repeated line,
               not a single token
"""

from __future__ import annotations

import re

STOP = set("""a an the i you we he she it they me him her us them my your our his its their this that these those
is are was were be been am do does did have has had will would can could shall should may might must
to of in on at for with from by as and or but if so not no yes please thank thanks it's i'm don't
""".split())
GENERIC_OK = {"yes", "no", "sure", "thanks", "thank", "sorry", "ok", "okay", "fine", "great", "good", "morning",
              "hello", "hi", "please", "certainly", "course", "alright", "maybe", "well"}
TURN = re.compile(r"^\s*([AB]):")


def words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']+", s.lower()) if w not in STOP and len(w) > 2}


def score(item, output: str) -> dict:
    opener = item.prompt.split("\n")[0][2:].strip()
    text = output.strip("\n")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    # the generation budget usually cuts the last line mid-sentence; don't judge it
    if len(lines) > 1 and not lines[-1].rstrip().endswith((".", "!", "?")):
        lines = lines[:-1]
    if not lines:
        return {"score": 0.0, "tiers": {"format": 0, "alternation": 0, "on_topic": 0, "non_degenerate": 0}, "failed": ["empty"]}
    reply = lines[0].strip()
    rest = lines[1:]
    turns = [TURN.match(ln) for ln in rest]
    fmt = float(bool(reply) and not TURN.match(reply) and all(turns))
    speakers = ["B"] + [m.group(1) for m in turns if m]
    alternation = float(all(speakers[i] != speakers[i + 1] for i in range(len(speakers) - 1)))
    ow, rw = words(opener), words(reply)
    generic = set(re.findall(r"[a-z']+", reply.lower())) & GENERIC_OK
    on_topic = float(bool(ow & rw) or bool(generic))
    non_degenerate = float(reply.lower() != opener.lower() and len(reply.split()) >= 2
                           and len(set(map(str.strip, lines))) == len(lines))
    tiers = {"format": fmt, "alternation": alternation, "on_topic": on_topic, "non_degenerate": non_degenerate}
    failed = [k for k, v in tiers.items() if v < 1]
    return {"score": sum(tiers.values()) / 4, "tiers": tiers, "failed": failed, "reply": reply}
