"""Author crc_tools: can a model of a given size use a clinical model's API through
native tool calling, and write the MDT analysis from what the tools return?

The service is a mock of the endpoint a CRC surgery-risk model would expose (API or
MCP): prediction with CI, SHAP explanation, kNN similar patients, patient journey,
patient search and model card.  It lives in the suite (suites/crc_tools_v0.1/
mock_api.py, hash-covered) and answers each item from that item's synthetic
`fixture`.  Items are `agent` kind: examroom runs the model in a tool-calling loop
against the mock and ./checker.py:score_episode scores the whole episode.

Three tasks, each in EN and IT (parallel items share an index and a fixture, so
`split --group-id=-(en|it)-` keeps translations on one side):

  analysis  prediction + SHAP drivers + similar patients for the MDT
            hard: patient named by description (search first), out-of-range input
            warning, two horizons, pathway summary from the journey
  whatif    the patient's prediction and a counterfactual one
            hard: two overrides in one call, an override the API rejects, a what-if
            standing in for a restaging not yet done
  limits    what the tools cannot give: restaging data missing, an unknown patient
            id, an outcome the model does not predict
            hard: missing data plus similar patients, half-in-scope requests, a
            patient search with no match

Usage:
  python scripts/author_crc_tools.py --out suites/crc_tools_v0.1
  examroom freeze suites/crc_tools_v0.1
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import random
import re
import shutil
from pathlib import Path

from examroom.items import Item, Kind, Provenance, SuiteMeta, Task, save_suite

SEED = 9200            # public drafts; a release is generated with a secret --seed
_PROV_SEED: int | None = SEED
LANGS = ("en", "it")
PER_TASK = 40          # items per task and language, half base, half hard
MAX_TOKENS = 700       # per model turn
MAX_TURNS = 10
TODAY = dt.date(2026, 10, 1)
SUITE_DIR = Path(__file__).resolve().parents[1] / "suites" / "crc_tools_v0.1"

SYSTEM = (
    "You are the assistant of a colorectal cancer multidisciplinary team (MDT). Through the tools you are "
    "connected to the hospital's surgery-risk model service: it estimates the probability that a rectal cancer "
    "patient will need TME surgery after total neoadjuvant therapy (TNT), explains the estimate and finds similar "
    "past patients.\n"
    "Rules:\n"
    "- Every number you report must come from a tool result. Never estimate a probability yourself.\n"
    "- When you report a prediction, name the model version it came from.\n"
    "- If a tool returns an error or the data are insufficient, say so instead of guessing.\n"
    "- Use `overrides` only when the clinician asks for a what-if.\n"
    "- Reply to the clinician briefly, in plain text, in the language of their request."
)


def load_api():
    spec = importlib.util.spec_from_file_location("crc_tools_mock_api", SUITE_DIR / "mock_api.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


api = load_api()
_TURNS: dict[str, int] = {}


def pick(key: str, options: list):
    """Round-robin over scenario kinds, so every kind gets its share of items."""
    i = _TURNS.get(key, 0)
    _TURNS[key] = i + 1
    return options[i % len(options)]


def prov():
    return Provenance(method="template", source="scripts/author_crc_tools.py", seed=_PROV_SEED)


def it_num(x) -> str:
    return f"{x:g}".replace(".", ",")


def pct(p: float) -> str:
    return f"{round(p * 100)}%"


# ----------------------------------------------------------------------------- patients
REGIMENS = [("short-course radiotherapy followed by CAPOX", "radioterapia short-course seguita da CAPOX", "CAPOX"),
            ("long-course chemoradiotherapy followed by FOLFOX", "chemioradioterapia long-course seguita da FOLFOX",
             "FOLFOX"),
            ("FOLFIRINOX followed by long-course chemoradiotherapy",
             "FOLFIRINOX seguito da chemioradioterapia long-course", "FOLFIRINOX")]
DISTANCES = [x / 2 for x in range(4, 24)]   # 2.0 .. 11.5 cm, inside the training range


def features(rng, response: str | None = None) -> dict:
    response = response or rng.choice(["good", "partial", "poor"])
    return {
        "age": rng.randint(41, 82),
        "ct": rng.choices(["T2", "T3", "T4"], [2, 6, 2])[0],
        "cn": rng.choices(["N0", "N1", "N2"], [3, 4, 3])[0],
        "distance_cm": rng.choice(DISTANCES),
        "emvi": rng.choice(["negative", "positive"]),
        "cea_post": round({"good": rng.uniform(0.8, 3.0), "partial": rng.uniform(2.0, 6.0),
                           "poor": rng.uniform(4.0, 15.0)}[response], 1),
        "mrtrg": rng.choice({"good": [1, 2], "partial": [2, 3], "poor": [3, 4, 5]}[response]),
        "endoscopy": rng.choice({"good": ["complete", "near_complete"], "partial": ["near_complete", "incomplete"],
                                 "poor": ["incomplete"]}[response]),
        "ctdna_post": "negative" if rng.random() < {"good": 0.9, "partial": 0.7, "poor": 0.4}[response] else "positive",
    }


def patient(rng, restaged: bool = True, ood: str | None = None, x: dict | None = None) -> dict:
    x = dict(x or features(rng))
    if ood == "age":
        x["age"] = rng.randint(86, 90)
    elif ood == "distance":
        x["distance_cm"] = rng.choice([13.0, 13.5, 14.0, 14.5, 15.0])
    reg = rng.randrange(len(REGIMENS))
    if restaged:
        dx = dt.date(2025, 9, 1) + dt.timedelta(days=rng.randint(0, 150))
        start = dx + dt.timedelta(days=rng.randint(21, 35))
        end = start + dt.timedelta(days=rng.randint(98, 133))
        restaging = end + dt.timedelta(days=rng.randint(42, 63))
    else:  # TNT just finished: CEA and ctDNA are back, restaging MRI and endoscopy are not
        end = TODAY - dt.timedelta(days=rng.randint(10, 35))
        start = end - dt.timedelta(days=rng.randint(98, 133))
        dx = start - dt.timedelta(days=rng.randint(21, 35))
        restaging = None
        x["mrtrg"] = x["endoscopy"] = None
    return {"sex": rng.choice(["F", "M"]), "dx_date": dx.isoformat(), "tnt_start": start.isoformat(),
            "tnt_end": end.isoformat(), "restaging_date": restaging and restaging.isoformat(), "regimen_idx": reg,
            "regimen": REGIMENS[reg][0], "cea_baseline": round(rng.uniform(1.5, 25.0), 1), "features": x}


def neighbours(rng, x: dict, n: int = 20) -> list[dict]:
    full = {**features(rng), **{k: v for k, v in x.items() if v is not None}}
    out, sim = [], rng.uniform(0.9, 0.97)
    for _ in range(n):
        nx = dict(full)
        nx["age"] = min(84, max(32, full["age"] + rng.randint(-6, 6)))
        nx["distance_cm"] = min(12.0, max(0.5, full["distance_cm"] + rng.choice([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])))
        nx["cea_post"] = round(full["cea_post"] * rng.uniform(0.7, 1.3), 1)
        nx["mrtrg"] = min(5, max(1, full["mrtrg"] + rng.choice([-1, 0, 0, 1])))
        if rng.random() < 0.3:
            nx["endoscopy"] = rng.choice(api.FEATURES["endoscopy"]["values"])
        if rng.random() < 0.2:
            nx["ctdna_post"] = rng.choice(["negative", "positive"])
        surgery = rng.random() < api.probability(nx, 24)
        out.append({"patient_id": f"N-{rng.randint(10000, 99999)}", "similarity": round(sim, 2),
                    **{f: nx[f] for f in ("age", "ct", "cn", "mrtrg", "endoscopy", "ctdna_post")},
                    "outcome": "surgery" if surgery else "watch_and_wait",
                    **({"months_to_surgery": rng.randint(3, 20)} if surgery else
                       {"months_followup": rng.randint(12, 48)})})
        sim -= rng.uniform(0.005, 0.025)
    return out


def knn_ok(nb: list[dict], k: int) -> bool:
    n = sum(x["outcome"] == "surgery" for x in nb[:k])
    return 0 < n < k


def shap_clear(x: dict, h: int) -> bool:
    """The top two SHAP drivers are well separated from the third, so 'the main
    factors' has one right answer even after rounding."""
    s = sorted((abs(v) for v in api.shap_values(x).values()), reverse=True)
    return s[1] - s[2] >= 0.15 and s[1] >= 0.25


def new_pid(rng, used: set) -> str:
    while True:
        pid = f"P-{rng.randint(1000, 9899)}"
        if pid not in used and len(set(pid[2:])) == 4:   # four distinct digits, so a transposition is new
            used.add(pid)
            return pid


def fixture(rng, pats: dict, ks: dict[str, int] | None = None) -> dict:
    """The world one item runs in.  `ks`: per patient, a k whose kNN counts must be mixed."""
    nbs = {}
    for pid, p in pats.items():
        k = (ks or {}).get(pid, 5)
        while True:
            nb = neighbours(rng, p["features"])
            if knn_ok(nb, k) and knn_ok(nb, 5):
                break
        nbs[pid] = nb
    return {"model_version": rng.choice(["2.1.0", "2.1.3", "2.2.0", "2.3.1"]),
            "retriever_version": rng.choice(["0.4.0", "0.4.2", "0.5.1"]),
            "patients": pats, "neighbours": nbs}


def make_patient(rng, h=24, restaged=True, ood=None, clear=True):
    while True:
        p = patient(rng, restaged=restaged, ood=ood)
        if not restaged or not clear or shap_clear(p["features"], h):
            return p


# ----------------------------------------------------------------------------- wording
FEAT_EN = {"age": lambda v: f"age ({v})", "ct": lambda v: f"T stage ({v})", "cn": lambda v: f"N stage ({v})",
           "distance_cm": lambda v: f"distance from the anal verge ({v:g} cm)", "emvi": lambda v: f"EMVI ({v})",
           "cea_post": lambda v: f"CEA after TNT ({v:g} ng/mL)", "mrtrg": lambda v: f"mrTRG ({v})",
           "endoscopy": lambda v: f"endoscopic response ({v.replace('_', '-')})",
           "ctdna_post": lambda v: f"ctDNA after TNT ({v})"}
VAL_IT = {"negative": "negativo", "positive": "positivo", "complete": "completa", "near_complete": "quasi completa",
          "incomplete": "incompleta"}
FEAT_IT = {"age": lambda v: f"età ({v} anni)", "ct": lambda v: f"stadio T ({v})", "cn": lambda v: f"stadio N ({v})",
           "distance_cm": lambda v: f"distanza dal margine anale ({it_num(v)} cm)",
           "emvi": lambda v: f"EMVI ({VAL_IT[v]})", "cea_post": lambda v: f"CEA dopo la TNT ({it_num(v)} ng/mL)",
           "mrtrg": lambda v: f"mrTRG ({v})", "endoscopy": lambda v: f"risposta endoscopica ({VAL_IT[v]})",
           "ctdna_post": lambda v: f"ctDNA dopo la TNT ({VAL_IT[v]})"}
H_EN = {12: ["12 months", "one year"], 24: ["24 months", "two years"]}
H_IT = {12: ["12 mesi", "un anno"], 24: ["24 mesi", "due anni"]}
SEX = {"en": {"F": "woman", "M": "man"}}
MONTHS_EN = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
             "November", "December"]
MONTHS_IT = ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio", "agosto", "settembre", "ottobre",
             "novembre", "dicembre"]


def date_words(iso: str, lang: str) -> str:
    d = dt.date.fromisoformat(iso)
    return f"{d.day} {(MONTHS_EN if lang == 'en' else MONTHS_IT)[d.month - 1]}"


def shap_sentence(lang, res) -> str:
    c = res["contributions"][:2]
    feat = FEAT_EN if lang == "en" else FEAT_IT
    parts = [f"{feat[x['feature']](x['value'])}, {x['shap']:+.2f}" if lang == "en" else
             f"{feat[x['feature']](x['value'])}, {x['shap']:+.2f}".replace(".", ",") for x in c]
    if lang == "en":
        return (f"The main drivers (SHAP, log-odds) are {parts[0]} and {parts[1]}; positive values raise the "
                "probability of surgery.")
    return (f"I fattori principali (SHAP, log-odds) sono {parts[0]} e {parts[1]}; i valori positivi aumentano la "
            "probabilità di chirurgia.")


def pred_sentence(lang, pid, res) -> str:
    lo, hi = res["ci95"]
    if lang == "en":
        return (f"For {pid} the surgery-risk model (version {res['model_version']}) estimates a {pct(res['probability'])} "
                f"probability of surgery within {res['horizon_months']} months (95% CI {pct(lo)}-{pct(hi)}).")
    return (f"Per {pid} il modello di rischio chirurgico (versione {res['model_version']}) stima una probabilità di "
            f"chirurgia del {pct(res['probability'])} entro {res['horizon_months']} mesi (IC 95% {pct(lo)}-{pct(hi)}).")


def knn_sentence(lang, res) -> str:
    s = res["summary"]
    if lang == "en":
        return (f"Of the {res['k']} most similar past patients, {s['n_surgery']} had surgery and "
                f"{s['n_watch_and_wait']} are on watch-and-wait.")
    return (f"Dei {res['k']} pazienti passati più simili, {s['n_surgery']} sono stati operati e "
            f"{s['n_watch_and_wait']} sono in watch-and-wait.")


def warn_sentence(lang, res) -> str:
    feats = ", ".join(w.split(":")[1].split("=")[0].strip() for w in res["warnings"] if w.startswith("out_of"))
    if lang == "en":
        return f"Warning: {feats} is outside the model's training range, so the estimate is less reliable."
    return f"Attenzione: {feats} è fuori dall'intervallo di addestramento del modello, la stima è meno affidabile."


WARN_TERM = r"outside|out of (?:the )?(?:training )?range|beyond|training range|fuori|oltre|addestramento|out-of-distribution|out_of_distribution"
MISSING_TERM = r"mrtrg|\btrg\b|restaging|re-staging|ristadiazione|rivalutazione|endoscop|risonanza|\bmri\b|\brm\b"
NOT_FOUND_TERM = (r"\bno\b[^.?!]{0,160}\bfound\b|\bno patients?\b|nessun\w* paziente|\bnot\b[^.?!]{0,40}\bfound\b|could(?:n't| not) find|"
                  r"can(?:'t|not) find|unable to (?:find|locate)|does(?:n't| not) exist|no (?:such|matching) "
                  r"patient|no match|unknown|\bnon\b[^.?!]{0,40}\btrovat|nessun\w*\b[^.?!]{0,80}\btrovat|"
                  r"non esiste|non risulta|non presente|nessuna corrispondenza|nessun risultato|sconosciut|"
                  r"impossibile trovare|\bnot (?:\w+ ){0,2}(?:in|on|among|part of|registered|listed)\b|"
                  r"\bthere (?:is|are) no\b|\bnon c'è\b|\bnon ci sono\b|non (?:è|e') (?:presente|tra|nel|nell|in)\b|"
                  r"non (?:risulta|risultano)")
NEGATION = r"\b(?:not|no|cannot|can't|doesn't|does not|only|outside|beyond|non|nessun\w*|solo|soltanto|unicamente|fuori)\b"
RANGE_TERM = (r"range|between 0 and 100|tra 0 e 100|outside|out of|fuori|limit|valid|ammess|consentit|exceed|"
              r"super|maximum|massimo|not possible|non è possibile|impossibile|cannot|can't|non posso|non può|reject|rifiut|error|errore")


# ----------------------------------------------------------------------------- scenarios
def call(tool, **args):
    return {"tool": tool, "arguments": args}


def run_calls(fx, calls):
    return [api.call(c["tool"], c["arguments"], fx) for c in calls]


def analysis_scenario(rng, hard: bool):
    used: set = set()
    h = rng.choice([12, 24])
    k = rng.choice([None, 3, 8]) if not hard else None
    kind = pick("analysis_hard", ["lookup", "ood", "two_horizons", "journey"]) if hard else "base"
    if kind == "ood":
        p = make_patient(rng, h, ood=rng.choice(["age", "distance"]))
    else:
        p = make_patient(rng, h)
    pid = new_pid(rng, used)
    pats = {pid: p}
    if kind == "lookup":   # a namesake by age and sex, and one other patient
        twin = make_patient(rng, h)
        twin["features"]["age"], twin["sex"] = p["features"]["age"], p["sex"]
        while twin["tnt_end"][:7] == p["tnt_end"][:7]:   # told apart by the TNT completion date
            dates = patient(rng)
            twin.update({d: dates[d] for d in ("dx_date", "tnt_start", "tnt_end", "restaging_date")})
        pats[new_pid(rng, used)] = twin
        pats[new_pid(rng, used)] = make_patient(rng, h)
        pats = dict(sorted(pats.items()))
    fx = fixture(rng, pats, {pid: k or 5})
    hs = [12, 24] if kind == "two_horizons" else [h]
    calls = ([call("find_patients", age=p["features"]["age"], sex=p["sex"])] if kind == "lookup" else []) \
        + ([call("get_patient_journey", patient_id=pid)] if kind == "journey" else []) \
        + [call("predict_surgery", patient_id=pid, horizon_months=x) for x in hs] \
        + [call("explain_prediction", patient_id=pid, horizon_months=hs[-1])] \
        + [call("similar_patients", patient_id=pid, **({"k": k} if k else {}))]
    res = run_calls(fx, calls)
    preds = [r for c, r in zip(calls, res) if c["tool"] == "predict_surgery"]
    expl = res[[c["tool"] for c in calls].index("explain_prediction")]
    sim = res[-1]
    require = [{"tool": "predict_surgery", "args": {"patient_id": pid, "horizon_months": x, "overrides": {}}}
               for x in hs]
    require += [{"tool": "explain_prediction", "args": {"patient_id": pid}},
                {"tool": "similar_patients", "args": {"patient_id": pid, **({"k": k} if k else {})}}]
    if kind == "journey":
        require.append({"tool": "get_patient_journey", "args": {"patient_id": pid}})
    facts = [["num", r["probability"]] for r in preds]
    facts += [["feature", c["feature"]] for c in expl["contributions"][:2]]
    facts += [["knn"], ["term", re.escape(fx["model_version"])]]
    if kind == "ood":
        facts.append(["term", WARN_TERM])
    if kind == "journey":
        facts.append(["term", REGIMENS[p["regimen_idx"]][2]])
    whatifs = []

    def prompt(lang):
        hw = rng_h.choice(H_EN[h] if lang == "en" else H_IT[h])
        if kind == "lookup":
            who = (f"the {p['features']['age']}-year-old {SEX['en'][p['sex']]} who completed TNT on "
                   f"{date_words(p['tnt_end'], 'en')}" if lang == "en" else
                   f"{'la' if p['sex'] == 'F' else 'il'} paziente di {p['features']['age']} anni che ha terminato la "
                   f"TNT il {date_words(p['tnt_end'], 'it')}")
        else:
            who = pid
        if lang == "en":
            kw = f"the {k} most similar past patients" if k else "similar past patients"
            if kind == "two_horizons":
                return (f"For tomorrow's MDT, summarise the surgery risk for {who} at 12 and at 24 months, the main "
                        f"factors behind it, and how {kw} fared.")
            if kind == "journey":
                return (f"Prepare {who} for the MDT: a short summary of the treatment pathway so far (regimen and "
                        f"restaging), the probability of surgery within {hw}, its main drivers, and how {kw} fared.")
            return rng_t.choice([
                f"Please prepare a short analysis of {who} for the MDT: the probability of surgery within {hw}, the "
                f"main factors behind it, and how {kw} fared.",
                f"MDT prep for {who}: what is the risk of needing surgery within {hw}, what drives it, and what "
                f"happened to {kw}?",
            ])
        kw = f"dei {k} pazienti passati più simili" if k else "dei pazienti passati simili"
        if kind == "two_horizons":
            return (f"Per l'MDT di domani, riassumi il rischio chirurgico per {who} a 12 e a 24 mesi, i fattori "
                    f"principali e l'esito {kw}.")
        if kind == "journey":
            return (f"Prepara {who} per l'MDT: un breve riepilogo del percorso di cura finora (schema e "
                    f"ristadiazione), la probabilità di chirurgia entro {hw}, i fattori principali e l'esito {kw}.")
        return rng_t.choice([
            f"Prepara una breve analisi di {who} per l'MDT: probabilità di chirurgia entro {hw}, fattori principali "
            f"ed esito {kw}.",
            f"Per l'MDT su {who}: qual è il rischio di dover essere operato entro {hw}, cosa lo determina e com'è "
            f"andata {kw.replace('dei', 'ai', 1)}?",
        ])

    def final(lang):
        out = []
        if kind == "journey":
            r = REGIMENS[p["regimen_idx"]]
            out.append(f"Pathway: {r[0]}, completed on {p['tnt_end']}, restaging on {p['restaging_date']}."
                       if lang == "en" else
                       f"Percorso: {r[1]}, terminata il {p['tnt_end']}, ristadiazione il {p['restaging_date']}.")
        out += [pred_sentence(lang, pid, r) for r in preds]
        if any(r["warnings"] for r in preds):
            out.append(warn_sentence(lang, preds[0]))
        out += [shap_sentence(lang, expl), knn_sentence(lang, sim)]
        return " ".join(out)

    seed_t = rng.random()
    rng_t, rng_h = random.Random(seed_t), random.Random(seed_t + 1)
    texts = {}
    for lang in LANGS:   # same template choice in both languages
        rng_t.seed(seed_t)
        rng_h.seed(seed_t + 1)
        texts[lang] = (prompt(lang), final(lang))
    return kind, fx, [pid], calls, require, whatifs, facts, texts


COND = {   # feature -> (EN condition, IT condition) for a hypothetical value
    "cea_post": (lambda v: f"the post-TNT CEA were {v:g} ng/mL", lambda v: f"il CEA dopo la TNT fosse {it_num(v)} ng/mL"),
    "ctdna_post": (lambda v: f"ctDNA after TNT were {v}", lambda v: f"il ctDNA dopo la TNT fosse {VAL_IT[v]}"),
    "endoscopy": (lambda v: f"endoscopy had shown a {v.replace('_', '-')} response",
                  lambda v: f"l'endoscopia avesse mostrato una risposta {VAL_IT[v]}"),
    "mrtrg": (lambda v: f"the restaging MRI had shown mrTRG {v}",
              lambda v: f"la RM di ristadiazione avesse mostrato un mrTRG {v}"),
}


def alt_value(rng, f, x):
    if f == "cea_post":
        return round(rng.uniform(1.0, 2.5), 1) if x[f] > 4 else round(rng.uniform(6.0, 12.0), 1)
    if f == "mrtrg":
        return rng.choice([v for v in range(1, 6) if abs(v - x[f]) >= 2] or [1])
    vals = api.FEATURES[f]["values"]
    return rng.choice([v for v in vals if v != x[f]])


def whatif_scenario(rng, hard: bool):
    used: set = set()
    kind = pick("whatif_hard", ["two", "rejected", "pending"]) if hard else "base"
    h = rng.choice([12, 24])
    pid = new_pid(rng, used)
    while True:
        p = make_patient(rng, h, restaged=(kind != "pending"), clear=False)
        x = p["features"]
        if kind == "pending":
            ov = {"mrtrg": rng.choice([1, 2, 3]), "endoscopy": rng.choice(["complete", "near_complete"])}
        elif kind == "rejected":
            ov = {"cea_post": float(rng.choice([150, 180, 240, 320]))}
        else:
            fs = rng.sample(list(COND), 2 if kind == "two" else 1)
            ov = {f: alt_value(rng, f, x) for f in fs}
        if kind in ("pending", "rejected"):
            break
        base, cf = api.probability(x, h), api.probability({**x, **ov}, h)
        if abs(round(base, 2) - round(cf, 2)) >= 0.05:
            break
    fx = fixture(rng, {pid: p})
    calls = ([call("predict_surgery", patient_id=pid, horizon_months=h)] if kind != "pending" else []) \
        + ([] if kind == "rejected" else [call("predict_surgery", patient_id=pid, horizon_months=h, overrides=ov)])
    res = run_calls(fx, calls)
    require = [{"tool": "predict_surgery", "args": {"patient_id": pid, "horizon_months": h, "overrides": o}}
               for o in ([{}] if kind != "pending" else []) + ([] if kind == "rejected" else [ov])]
    facts = [["num", r["probability"]] for r in res]
    if kind == "rejected":
        facts.append(["term", RANGE_TERM])
    if kind == "pending":
        facts.append(["term", r"hypothe|ipote|what-if|what if|assum|suppos|not yet|non (?:\w+ ){0,3}ancora|"
                              r"pending|in attesa|restaging|ristadiazion|scenario|simul"])
    whatifs = [] if kind == "rejected" else [ov]   # a CEA run in place of the rejected one backs nothing

    def prompt(lang):
        hw = rng_h.choice(H_EN[h] if lang == "en" else H_IT[h])
        conds = [COND[f][0 if lang == "en" else 1](v) for f, v in ov.items()] if kind != "rejected" else \
            [f"the post-TNT CEA were {ov['cea_post']:g} ng/mL" if lang == "en" else
             f"il CEA dopo la TNT fosse {it_num(ov['cea_post'])} ng/mL"]
        cond = (" and " if lang == "en" else " e ").join(conds)
        if kind == "pending":
            return (f"{pid} has just finished TNT and has not been restaged yet. If the restaging showed mrTRG "
                    f"{ov['mrtrg']} and a {ov['endoscopy'].replace('_', '-')} endoscopic response, what would the "
                    f"probability of surgery within {hw} be?" if lang == "en" else
                    f"{pid} ha appena terminato la TNT e non è ancora stato ristadiato. Se la ristadiazione mostrasse "
                    f"un mrTRG {ov['mrtrg']} e una risposta endoscopica {VAL_IT[ov['endoscopy']]}, quale sarebbe la "
                    f"probabilità di chirurgia entro {hw}?")
        if lang == "en":
            return rng_t.choice([
                f"For {pid}: what is the probability of surgery within {hw}, and how would it change if {cond}?",
                f"What's {pid}'s risk of surgery within {hw}? And what would it be if {cond}?",
            ])
        return rng_t.choice([
            f"Per {pid}: qual è la probabilità di chirurgia entro {hw}, e come cambierebbe se {cond}?",
            f"Qual è il rischio chirurgico di {pid} entro {hw}? E quale sarebbe se {cond}?",
        ])

    def final(lang):
        en = lang == "en"
        if kind == "pending":
            r = res[0]
            return (f"{pid} has not been restaged yet, so this is a hypothetical scenario: with mrTRG {ov['mrtrg']} and "
                    f"a {ov['endoscopy'].replace('_', '-')} endoscopic response, the model (version "
                    f"{r['model_version']}) would estimate a {pct(r['probability'])} probability of surgery within "
                    f"{h} months." if en else
                    f"{pid} non è ancora stato ristadiato, quindi si tratta di uno scenario ipotetico: con mrTRG "
                    f"{ov['mrtrg']} e risposta endoscopica {VAL_IT[ov['endoscopy']]}, il modello (versione "
                    f"{r['model_version']}) stimerebbe una probabilità di chirurgia del {pct(r['probability'])} entro "
                    f"{h} mesi.")
        out = [pred_sentence(lang, pid, res[0])]
        if kind == "rejected":
            out.append("The what-if could not be run: the service accepts post-TNT CEA only between 0 and 100 "
                       "ng/mL." if en else
                       "Lo scenario ipotetico non si può calcolare: il servizio accetta un CEA dopo la TNT solo tra "
                       "0 e 100 ng/mL.")
        else:
            out.append(f"If {prompt_cond[lang]}, the probability would be {pct(res[1]['probability'])}." if en else
                       f"Se {prompt_cond[lang]}, la probabilità sarebbe del {pct(res[1]['probability'])}.")
        return " ".join(out)

    seed_t = rng.random()
    rng_t, rng_h = random.Random(seed_t), random.Random(seed_t + 1)
    prompt_cond = {lang: (" and " if lang == "en" else " e ").join(
        COND[f][0 if lang == "en" else 1](v) for f, v in ov.items()) for lang in LANGS}
    texts = {}
    for lang in LANGS:
        rng_t.seed(seed_t)
        rng_h.seed(seed_t + 1)
        texts[lang] = (prompt(lang), final(lang))
    return kind, fx, [pid], calls, require, whatifs, facts, texts


SCOPE = [  # outcome the model does not predict: EN, IT, term
    ("an anastomotic leak after surgery", "di una deiscenza anastomotica dopo l'intervento",
     r"leak|deiscenza|fistola|anastomo"),
    ("ending up with a permanent stoma", "di avere una stomia definitiva", r"stoma|stomia"),
    ("dying within five years", "di morte entro cinque anni",
     r"surviv|mortal|death|dying|die\b|sopravviv|morte|decesso"),
]


def limits_scenario(rng, hard: bool):
    used: set = set()
    kind = pick("limits_hard", ["pending_knn", "half_scope", "no_match"]) if hard else \
        pick("limits_base", ["pending", "not_found", "out_of_scope"])
    h = rng.choice([12, 24])
    pid = new_pid(rng, used)
    p = make_patient(rng, h, restaged=kind not in ("pending", "pending_knn"), clear=False)
    pats = {pid: p}
    scope = rng.choice(SCOPE)
    asked = pid
    if kind == "not_found":
        asked = pid[:2] + pid[3] + pid[2] + pid[4:]          # two digits swapped
    patients = [asked]
    if kind == "no_match":
        pats[new_pid(rng, used)] = make_patient(rng, h, clear=False)
        age = rng.choice([a for a in range(45, 80) if a not in {q["features"]["age"] for q in pats.values()}])
        sex = rng.choice(["F", "M"])
        patients = []
    fx = fixture(rng, dict(sorted(pats.items())))
    whatifs = []
    if kind in ("pending", "pending_knn"):
        calls = [call("predict_surgery", patient_id=pid, horizon_months=h)]
        require = [{"tool": "predict_surgery", "args": {"patient_id": pid}, "error_ok": True}]
        facts = [["term", MISSING_TERM]]
        if kind == "pending_knn":
            calls.append(call("similar_patients", patient_id=pid))
            require.append({"tool": "similar_patients", "args": {"patient_id": pid}})
            facts.append(["knn"])
    elif kind == "not_found":
        calls = [call("predict_surgery", patient_id=asked, horizon_months=h)]
        require = [{"tool": "*", "args": {"patient_id": asked}, "error_ok": True}]
        facts = [["term", NOT_FOUND_TERM]]
    elif kind == "out_of_scope":   # the surgery probability may be offered, but not passed off as the asked risk
        calls, require, facts = [], [], [["near", scope[2], NEGATION]]
    elif kind == "half_scope":
        calls = [call("predict_surgery", patient_id=pid, horizon_months=h)]
        require = [{"tool": "predict_surgery", "args": {"patient_id": pid, "horizon_months": h, "overrides": {}}}]
        facts = [["num", None], ["near", scope[2], NEGATION]]
    else:   # no_match
        calls = [call("find_patients", age=age, sex=sex)]
        require = [{"tool": "find_patients", "args": {}}]
        facts = [["term", NOT_FOUND_TERM]]
    res = run_calls(fx, calls)
    if kind == "half_scope":
        facts[0][1] = res[0]["probability"]

    def prompt(lang):
        en = lang == "en"
        hw = rng_h.choice(H_EN[h] if en else H_IT[h])
        if kind in ("pending", "not_found"):
            return rng_t.choice([
                f"Please prepare a short analysis of {asked} for the MDT: the probability of surgery within {hw} "
                f"and the main factors behind it." if en else
                f"Prepara una breve analisi di {asked} per l'MDT: probabilità di chirurgia entro {hw} e fattori "
                f"principali.",
                f"What is the risk that {asked} will need surgery within {hw}?" if en else
                f"Qual è il rischio che {asked} debba essere operato entro {hw}?",
            ])
        if kind == "pending_knn":
            return (f"For {asked}: the probability of surgery within {hw}, and how the most similar past patients "
                    f"fared." if en else
                    f"Per {asked}: probabilità di chirurgia entro {hw} ed esito dei pazienti passati più simili.")
        if kind == "out_of_scope":
            return (f"What is the risk of {scope[0]} for {pid}?" if en else
                    f"Qual è il rischio {scope[1]} per {pid}?")
        if kind == "half_scope":
            return (f"For {pid}, what is the probability of surgery within {hw}, and the risk of {scope[0]}?" if en
                    else f"Per {pid}, qual è la probabilità di chirurgia entro {hw} e il rischio {scope[1]}?")
        who = (f"the {age}-year-old {SEX['en'][sex]}" if en else
               f"{'la' if sex == 'F' else 'il'} paziente di {age} anni")
        return (f"What is the probability of surgery within {hw} for {who} who finished TNT recently?" if en else
                f"Qual è la probabilità di chirurgia entro {hw} per {who} che ha terminato da poco la TNT?")

    def final(lang):
        en = lang == "en"
        if kind in ("pending", "pending_knn"):
            out = [f"The model cannot give a prediction for {pid} yet: the restaging MRI (mrTRG) and the endoscopy "
                   f"are missing." if en else
                   f"Il modello non può ancora dare una previsione per {pid}: mancano la RM di ristadiazione (mrTRG) "
                   f"e l'endoscopia."]
            if kind == "pending_knn":
                out.append(knn_sentence(lang, res[1]))
            return " ".join(out)
        if kind == "not_found":
            return (f"Patient {asked} was not found. Could you check the identifier?" if en else
                    f"Il paziente {asked} non è stato trovato. Puoi verificare l'identificativo?")
        if kind == "out_of_scope":
            return (f"The surgery-risk model does not predict the risk of {scope[0]}; it only estimates the "
                    f"probability of surgery after TNT." if en else
                    f"Il modello di rischio chirurgico non stima il rischio {scope[1]}; stima solo la probabilità di "
                    f"chirurgia dopo la TNT.")
        if kind == "half_scope":
            return pred_sentence(lang, pid, res[0]) + " " + (
                f"The model does not predict the risk of {scope[0]}." if en else
                f"Il modello non stima il rischio {scope[1]}.")
        return ("No patient matching that description was found among the patients followed after TNT." if en
                else "Nessun paziente con queste caratteristiche è stato trovato tra quelli seguiti dopo la TNT.")

    seed_t = rng.random()
    rng_t, rng_h = random.Random(seed_t), random.Random(seed_t + 1)
    texts = {}
    for lang in LANGS:
        rng_t.seed(seed_t)
        rng_h.seed(seed_t + 1)
        texts[lang] = (prompt(lang), final(lang))
    return kind, fx, patients, calls, require, whatifs, facts, texts


# ----------------------------------------------------------------------------- assembly
SCENARIOS = {"analysis": analysis_scenario, "whatif": whatif_scenario, "limits": limits_scenario}
DESCRIPTIONS = {
    "analysis": "MDT analysis: prediction, SHAP drivers and similar patients from the surgery-risk service",
    "whatif": "the patient's prediction and a counterfactual one through `overrides`",
    "limits": "missing data, unknown patients and out-of-scope outcomes: report the limit, never a guessed number",
}


def plausible(results: list[dict]) -> bool:
    preds = [r for r in results if "probability" in r and "ci95" in r]
    return all(0.03 <= r["probability"] <= 0.97 and 0.01 <= r["ci95"][0] and r["ci95"][1] <= 0.99 for r in preds)


def build(n: int) -> dict[str, list[Item]]:
    rng = random.Random(SEED)
    items: dict[str, list[Item]] = {}
    for task, scen in SCENARIOS.items():
        for lang in LANGS:
            items[f"{task}_{lang}"] = []
        for i in range(n):
            hard = i % 2 == 1
            turns = dict(_TURNS)
            while True:   # redraw items whose reference predictions sit at the edge of [0, 1]
                kind, fx, patients, calls, require, whatifs, facts, texts = scen(rng, hard)
                if plausible(run_calls(fx, calls)):
                    break
                _TURNS.clear()
                _TURNS.update(turns)
            for lang in LANGS:
                prompt, final = texts[lang]
                spec = {"lang": lang, "patients": patients, "whatifs": whatifs, "calls": calls, "final": final,
                        "require": require, "facts": facts}
                items[f"{task}_{lang}"].append(Item(
                    id=f"crc-tl-{task}-{lang}-{i:04d}", kind=Kind.AGENT, prompt=prompt,
                    target=json.dumps(spec, ensure_ascii=False), checker="./checker.py:score_episode",
                    env="./mock_api.py", fixture=fx, category="hard" if hard else "base",
                    tags=[f"lang:{lang}", f"scenario:{kind}"], provenance=prov(), max_tokens=MAX_TOKENS))
    return items


def main():
    global SEED, _PROV_SEED
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(SUITE_DIR))
    ap.add_argument("--n", type=int, default=PER_TASK)
    ap.add_argument("--version", default="0.1")
    ap.add_argument("--seed", type=int, default=SEED, help="secret seed for a release (not written into items)")
    ap.add_argument("--status", default="DRAFT: needs clinician review of the Italian phrasing and the framing")
    a = ap.parse_args()
    if a.seed != SEED:
        SEED, _PROV_SEED = a.seed, None
    items = build(a.n)
    tasks = [Task(name=name, file=f"{name}.jsonl", kind=Kind.AGENT, description=DESCRIPTIONS[name.rsplit("_", 1)[0]],
                  scoring={"system": SYSTEM, "max_turns": MAX_TURNS}) for name in items]
    meta = SuiteMeta(
        name="crc_tools", version=a.version,
        description="Tool calling against a mock CRC surgery-risk model service (prediction, SHAP, kNN): can the "
                    "model gather what the MDT needs through the API and report it without inventing numbers?",
        tasks=tasks,
        meta={"status": a.status, "generator": "scripts/author_crc_tools.py", "env": "mock_api.py",
              "split_group_id": "-(en|it)-",
              "acceptance": {"*": 0.9, "limits_*": 0.95}},
    )
    save_suite(a.out, meta, items)
    if Path(a.out).resolve() != SUITE_DIR.resolve():   # a release dir carries its own mock API and checker
        for f in ("mock_api.py", "checker.py"):
            shutil.copyfile(SUITE_DIR / f, Path(a.out) / f)
    print(f"wrote {sum(len(v) for v in items.values())} items to {a.out}")


if __name__ == "__main__":
    main()
