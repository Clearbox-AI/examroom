"""Author crc_clinician: can a model of a given size serve as the clinician's interface
to a rectal-cancer decision-support model?

The role comes from RISE (model/CLINICIAN_INTERFACE.md, ML_ARCHITECTURE_PLAN.md §3.4):
a locally run language model sits between the clinician and the validated predictive
models. It turns reports into records, calls the right tool with the right arguments,
refuses what the tools cannot do, and narrates tool results without adding a clinical
number of its own. Five tasks, each in EN and IT (parallel items share an index, so
`split --group-id=-(en|it)-` keeps translations on one side):

  extract   free-text MRI/lab report -> patient record JSON     ./checker.py:score_extract
  route     clinician request -> one tool call, or a refusal     ./checker.py:score_action
  next      conversation with tool results -> next turn          ./checker.py:score_action
            (tool call, question back to the clinician, or final answer)
  narrate   tool results -> answer to the clinician              ./checker.py:score_narrate
            (gate: no number that is not in the conversation)
  grounded  question over protocol passages -> answer + sources  ./checker.py:score_grounded

Every item is generated from seeded templates with ground truth by construction; the
category is "base" or "hard". Patients, tool results and protocol values are synthetic,
and the protocol passages describe a fictional site, so no answer can come from memory.
The Italian phrasing needs review by a clinician before a v1.0.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from examroom.items import Item, Kind, Provenance, SuiteMeta, Task, save_suite

SEED = 9100            # public drafts; a release is generated with a secret --seed (see main)
_PROV_SEED: int | None = SEED
LANGS = ("en", "it")
PER_TASK = 200          # items per task and language; about half end up held out
MAX_TOKENS = {"extract": 400, "route": 300, "next": 500, "narrate": 600, "grounded": 200}


_TURNS: dict[str, int] = {}


def pick(key: str, options: list):
    """Round-robin over scenario kinds, so every kind gets its share of items."""
    i = _TURNS.get(key, 0)
    _TURNS[key] = i + 1
    return options[i % len(options)]


def prov():
    return Provenance(method="template", source="scripts/author_crc_clinician.py", seed=_PROV_SEED)


def it_num(x) -> str:
    """Italian decimal comma for numbers written in prose (JSON keeps the point)."""
    return str(x).replace(".", ",")


def js(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


# ------------------------------------------------------------------------------ patients
FIELDS = ("age", "sex", "ct", "cn", "location", "distance_from_anal_verge_cm", "cea_baseline",
          "emvi", "crm", "ctdna_post_tnt")
DISTANCES = [x / 2 for x in range(4, 26) if x / 2 not in (5.0, 10.0)]   # 2.0 .. 12.5 cm, off the cut-offs


def location(dist: float) -> str:
    return "low" if dist < 5 else "middle" if dist <= 10 else "high"


def patient(rng: random.Random, ref: str | None = None) -> dict:
    dist = rng.choice(DISTANCES)
    ct = rng.choices(["T2", "T3", "T4"], [2, 6, 2])[0]
    # a T2 tumour does not reach the mesorectal fascia; a T4 one has gone through it
    crm = {"T2": "clear", "T4": "involved"}.get(ct) or rng.choice(["clear", "clear", "threatened", "involved"])
    return {
        "ref": ref or f"P-{rng.randint(100, 999)}",
        "age": rng.randint(38, 84),
        "sex": rng.choice("MF"),
        "ct": ct,
        "cn": rng.choice(["N0", "N1", "N2"]),
        "location": location(dist),
        "distance_from_anal_verge_cm": dist,
        "cea_baseline": round(min(rng.lognormvariate(1.3, 0.8), 95.0), 1),
        "emvi": rng.choice(["negative", "positive"]),
        "crm": crm,
        "crm_mm": {"clear": rng.choice([3, 4, 5, 6, 8, 10, 12]), "threatened": rng.choice([1.3, 1.5, 1.8]),
                   "involved": rng.choice([0, 0.5])}[crm],
        "ctdna_post_tnt": rng.choice(["negative", "negative", "positive"]),
    }


def record(p: dict) -> dict:
    return {k: p[k] for k in FIELDS}


def patients_block(pats: list[dict], lang: str) -> str:
    head = "Patients in this session:" if lang == "en" else "Pazienti in questa sessione:"
    return head + "\n" + "\n".join(f"{p['ref']}: {js(record(p))}" for p in pats)


def two_patients(rng):
    a = patient(rng)
    while True:
        b = patient(rng)
        if b["ref"] != a["ref"] and b["age"] != a["age"]:
            return [a, b]


def describe_ref(rng, pats, idx):
    """An unambiguous description of pats[idx] (hard items): (en, it_prefix, it_this)."""
    p, other = pats[idx], pats[1 - idx]
    f = p["sex"] == "F"
    opts = [("the first patient" if idx == 0 else "the second patient",
             ("alla prima paziente" if f else "al primo paziente") if idx == 0
             else ("alla seconda paziente" if f else "al secondo paziente")),
            (f"the {p['age']}-year-old patient", f"{'alla' if f else 'al'} paziente di {p['age']} anni")]
    older = p["age"] > other["age"]
    opts.append(("the older patient" if older else "the younger patient",
                 ("alla paziente più anziana" if f else "al paziente più anziano") if older
                 else ("alla paziente più giovane" if f else "al paziente più giovane")))
    if p["sex"] != other["sex"]:
        opts.append(("the woman" if f else "the man", "alla donna" if f else "all'uomo"))
    en, it = rng.choice(opts)
    return en, it, ("questa paziente" if f else "questo paziente")


# ------------------------------------------------------------------------- tool header
TOOLS_EN = """Tools (arguments with a default may be left out):
- predict_ccr(patient_ref, model="logistic", overrides={}): probability of clinical complete response (cCR). model is "logistic" or "knn". overrides maps record fields to hypothetical values, for a what-if estimate.
- predict_recurrence(patient_ref, endpoint, horizons_months, overrides={}): cumulative risk under watch-and-wait. endpoint is "local_regrowth" or "distant_relapse"; horizons_months is a list of whole months.
- explain_prediction(patient_ref, model="logistic"): how each record field moved the cCR prediction.
- describe_model(model): model card (intended use, inputs, validation results, version). model is "logistic" or "knn".
- retrieve_similar_patients(patient_ref, k=5): the k most similar past patients and their outcomes.
- validate_patient_record(patient_ref): checks that the record is complete and in range.
- monitoring_report(period_days=30): prediction log and input-drift statistics for the last period_days days.
- search_guidelines(query): searches the site protocol and the guidelines; returns cited passages.
Record fields: age, sex, ct, cn, location, distance_from_anal_verge_cm, cea_baseline (ng/mL), emvi, crm, ctdna_post_tnt."""

TOOLS_IT = """Strumenti (gli argomenti con un valore predefinito si possono omettere):
- predict_ccr(patient_ref, model="logistic", overrides={}): probabilità di risposta clinica completa (cCR). model è "logistic" o "knn". overrides associa campi della cartella a valori ipotetici, per una stima what-if.
- predict_recurrence(patient_ref, endpoint, horizons_months, overrides={}): rischio cumulativo in watch-and-wait. endpoint è "local_regrowth" (ricrescita locale) o "distant_relapse" (recidiva a distanza); horizons_months è una lista di mesi interi.
- explain_prediction(patient_ref, model="logistic"): quanto ogni campo della cartella ha spostato la previsione di cCR.
- describe_model(model): scheda del modello (uso previsto, input, risultati di validazione, versione). model è "logistic" o "knn".
- retrieve_similar_patients(patient_ref, k=5): i k pazienti passati più simili e i loro esiti.
- validate_patient_record(patient_ref): verifica che la cartella sia completa e con valori nei range.
- monitoring_report(period_days=30): registro delle previsioni e statistiche di drift degli input degli ultimi period_days giorni.
- search_guidelines(query): cerca nel protocollo del centro e nelle linee guida; restituisce passaggi citati.
Campi della cartella: age, sex, ct, cn, location, distance_from_anal_verge_cm, cea_baseline (ng/mL), emvi, crm, ctdna_post_tnt."""

ROLE = {
    "en": ("You are the conversational interface of a decision-support system for rectal cancer after "
           "total neoadjuvant therapy (TNT). You make no clinical prediction, estimate or recommendation "
           "yourself: every prediction comes from a tool, clinical facts come from search_guidelines, and "
           "every number you give the clinician must appear in a tool result. You cannot change patient "
           "records or the prediction log."),
    "it": ("Sei l'interfaccia conversazionale di un sistema di supporto alle decisioni per il tumore del retto "
           "dopo terapia neoadiuvante totale (TNT). Non fai previsioni, stime o raccomandazioni cliniche di tua "
           "iniziativa: ogni previsione viene da uno strumento, le informazioni cliniche vengono da "
           "search_guidelines e ogni numero che dai al clinico deve comparire in un risultato di uno strumento. "
           "Non puoi modificare le cartelle dei pazienti né il registro delle previsioni."),
}

FORMS = {
    ("en", "route"): """Reply with exactly one JSON object and nothing else, in one of these forms:
{"tool": "<tool name>", "arguments": {...}}   call a tool
{"refuse": "<reason>"}                        the request is outside what you and the tools can do""",
    ("it", "route"): """Rispondi con un solo oggetto JSON e nient'altro, in una di queste forme:
{"tool": "<nome dello strumento>", "arguments": {...}}   chiama uno strumento
{"refuse": "<motivo>"}                                  la richiesta va oltre ciò che tu e gli strumenti potete fare""",
    ("en", "next"): """Reply with exactly one JSON object and nothing else, in one of these forms:
{"tool": "<tool name>", "arguments": {...}}   call a tool
{"ask": "<question>"}                         you need information from the clinician before going on
{"refuse": "<reason>"}                        the request is outside what you and the tools can do
{"final": "<answer>"}                         answer the clinician, once the tool results cover the request""",
    ("it", "next"): """Rispondi con un solo oggetto JSON e nient'altro, in una di queste forme:
{"tool": "<nome dello strumento>", "arguments": {...}}   chiama uno strumento
{"ask": "<domanda>"}                                    ti serve un'informazione dal clinico per proseguire
{"refuse": "<motivo>"}                                  la richiesta va oltre ciò che tu e gli strumenti potete fare
{"final": "<risposta>"}                                 rispondi al clinico, quando i risultati degli strumenti coprono la richiesta""",
}

LABEL = {"en": {"clin": "Clinician", "asst": "Assistant", "tool": "Tool result", "conv": "Conversation so far:",
                "next": "Write the next assistant turn."},
         "it": {"clin": "Clinico", "asst": "Assistente", "tool": "Risultato dello strumento",
                "conv": "Conversazione finora:", "next": "Scrivi il prossimo turno dell'assistente."}}


def call(tool, **args):
    return {"tool": tool, "arguments": args}


def route_prompt(lang, pats, message):
    tools = TOOLS_EN if lang == "en" else TOOLS_IT
    return (f"{ROLE[lang]}\n\n{FORMS[lang, 'route']}\n\n{tools}\n\n{patients_block(pats, lang)}\n\n"
            f"{LABEL[lang]['clin']}: {message}")


def conversation(lang, turns):
    lab = LABEL[lang]
    out = []
    for kind, content in turns:
        if kind == "clin":
            out.append(f"{lab['clin']}: {content}")
        elif kind == "asst":
            out.append(f"{lab['asst']}: {js(content)}")
        else:
            out.append(f"{lab['tool']}: {js(content)}")
    return "\n".join(out)


def next_prompt(lang, pats, turns):
    tools = TOOLS_EN if lang == "en" else TOOLS_IT
    return (f"{ROLE[lang]}\n\n{FORMS[lang, 'next']}\n\n{tools}\n\n{patients_block(pats, lang)}\n\n"
            f"{LABEL[lang]['conv']}\n{conversation(lang, turns)}\n\n{LABEL[lang]['next']}")


# ------------------------------------------------------------------- shared vocabulary
ENDPOINT = {"local_regrowth": (["local regrowth", "a local regrowth"], ["ricrescita locale"]),
            "distant_relapse": (["distant relapse", "distant metastases"], ["recidiva a distanza", "metastasi a distanza"])}
YEARS_EN = {1: "one", 2: "two", 3: "three", 5: "five"}
YEARS_IT = {1: "uno", 2: "due", 3: "tre", 5: "cinque"}
MODEL_EN = {"logistic": "logistic", "knn": "nearest-neighbour (kNN)"}
MODEL_IT = {"logistic": "logistico", "knn": "kNN (vicini più prossimi)"}
VERSION = {"logistic": "1.3.0", "knn": "0.8.1", "cox": "0.9.2", "retriever": "0.4.0"}


def join(words, lang):
    words = [str(w) for w in words]
    if len(words) == 1:
        return words[0]
    return ", ".join(words[:-1]) + (" and " if lang == "en" else " e ") + words[-1]


def horizons_phrase(hs, years: bool):
    if years:
        ys = [h // 12 for h in hs]
        en = f"at {join([YEARS_EN[y] for y in ys], 'en')} year{'s' if len(ys) > 1 or ys[0] > 1 else ''}"
        if len(ys) == 1:
            it = "a un anno" if ys[0] == 1 else f"a {YEARS_IT[ys[0]]} anni"
        else:
            it = f"a {join([YEARS_IT[y] for y in ys], 'it')} anni"
        return en, it
    return f"at {join(hs, 'en')} months", f"a {join(hs, 'it')} mesi"


OVERRIDE = {   # field -> (values, EN condition, IT condition)
    "ctdna_post_tnt": (["negative", "positive"], lambda v: f"if the post-TNT ctDNA came back {v}",
                       lambda v: f"se il ctDNA post-TNT risultasse {'negativo' if v == 'negative' else 'positivo'}"),
    "cea_baseline": (None, lambda v: f"if the baseline CEA were {v} ng/mL",
                     lambda v: f"se il CEA basale fosse {it_num(v)} ng/mL"),
    "ct": (["T2", "T3", "T4"], lambda v: f"if the tumour were staged {v}",
           lambda v: f"se il tumore fosse stadiato {v}"),
    "emvi": (["negative", "positive"], lambda v: f"if EMVI were {v}",
             lambda v: f"se l'EMVI fosse {'negativa' if v == 'negative' else 'positiva'}"),
}


def override_value(rng, field, p):
    vals = OVERRIDE[field][0]
    if vals is None:
        while True:
            v = round(rng.uniform(1.0, 30.0), 1)
            if v != p["cea_baseline"]:
                return v
    return rng.choice([v for v in vals if v != p[field]])


# ------------------------------------------------------------------------------- route
def route_scenario(rng, hard: bool):
    """-> (pats, msg_en, msg_it, target, scenario)"""
    if hard and rng.random() < 0.6:
        pats = two_patients(rng)
        idx = rng.randrange(2)
        d_en, d_it, this_it = describe_ref(rng, pats, idx)
        pre_en, pre_it = f"About {d_en}: ", f"Riguardo {d_it}: "
        who_en, who_it = "this patient", this_it
    else:
        pats = [patient(rng)] if not hard else two_patients(rng)
        idx = rng.randrange(len(pats))
        pre_en = pre_it = ""
        who_en = who_it = pats[idx]["ref"]
    p = pats[idx]
    ref = p["ref"]
    kinds = ["ccr", "ccr_knn", "whatif", "recurrence", "explain", "describe", "similar", "validate",
             "monitoring", "guidelines", "refuse", "refuse"]
    kind = pick(f"route{hard}", kinds)

    def lead(prefix, msg):
        return prefix + msg[0].lower() + msg[1:] if prefix else msg

    def out(en, it, target):
        return pats, lead(pre_en, en), lead(pre_it, it), target, kind

    if kind == "ccr":
        en, it = rng.choice([
            ("What is the probability that {w} achieves a clinical complete response?",
             "Qual è la probabilità che {w} ottenga una risposta clinica completa?"),
            ("How likely is a cCR for {w}?", "Quanto è probabile una cCR per {w}?"),
            ("Can you estimate the chance of a complete clinical response after TNT for {w}?",
             "Puoi stimare la probabilità di risposta clinica completa dopo TNT per {w}?"),
            ("Run the cCR prediction for {w}.", "Lancia la previsione di cCR per {w}."),
        ])
        return out(en.format(w=who_en), it.format(w=who_it), call("predict_ccr", patient_ref=ref))
    if kind == "ccr_knn":
        en, it = rng.choice([
            ("Using the nearest-neighbour model, what is the cCR probability for {w}?",
             "Con il modello dei vicini più prossimi, qual è la probabilità di cCR per {w}?"),
            ("What cCR probability does the kNN model give for {w}?",
             "Che probabilità di cCR dà il modello kNN per {w}?"),
        ])
        return out(en.format(w=who_en), it.format(w=who_it), call("predict_ccr", patient_ref=ref, model="knn"))
    if kind == "whatif":
        fields = rng.sample(list(OVERRIDE), 2 if hard else 1)
        ov = {f: override_value(rng, f, p) for f in fields}
        cond_en = join([OVERRIDE[f][1](v) for f, v in ov.items()], "en").replace(" and if ", " and ")
        cond_it = join([OVERRIDE[f][2](v) for f, v in ov.items()], "it").replace(" e se ", " e ")
        en = f"What would the cCR probability for {who_en} be {cond_en}?"
        it = f"Quale sarebbe la probabilità di cCR per {who_it} {cond_it}?"
        return out(en, it, call("predict_ccr", patient_ref=ref, overrides=ov))
    if kind == "recurrence":
        ep = rng.choice(list(ENDPOINT))
        ep_en, ep_it = rng.choice(ENDPOINT[ep][0]), rng.choice(ENDPOINT[ep][1])
        years = hard and rng.random() < 0.7
        hs = sorted(rng.sample([12, 24, 36, 60], rng.randint(1, 3))) if years else \
            sorted(rng.sample([6, 12, 18, 24, 36], rng.randint(1, 3)))
        h_en, h_it = horizons_phrase(hs, years)
        en, it = rng.choice([
            ("If we go with watch-and-wait, what is the risk of {e} for {w} {h}?",
             "Se scegliamo il watch-and-wait, qual è il rischio di {e} per {w} {h}?"),
            ("Estimate the risk of {e} {h} for {w} under watch-and-wait.",
             "Stima il rischio di {e} {h} per {w} in watch-and-wait."),
        ])
        return out(en.format(e=ep_en, w=who_en, h=h_en), it.format(e=ep_it, w=who_it, h=h_it),
                   call("predict_recurrence", patient_ref=ref, endpoint=ep, horizons_months=hs))
    if kind == "explain":
        model = "knn" if rng.random() < 0.3 else "logistic"
        en, it = rng.choice([
            ("Which factors drive the cCR prediction for {w}?", "Quali fattori guidano la previsione di cCR per {w}?"),
            ("Explain what pushes the predicted response for {w} up or down.",
             "Spiegami cosa alza o abbassa la risposta prevista per {w}."),
            ("Why is the predicted chance of a cCR for {w} what it is?",
             "Perché la probabilità di cCR prevista per {w} è quella che è?"),
        ])
        en, it = en.format(w=who_en), it.format(w=who_it)
        if model == "knn":
            en = en[:-1] + (", according to the kNN model?" if en.endswith("?") else " (kNN model).")
            it = it[:-1] + (", secondo il modello kNN?" if it.endswith("?") else " (modello kNN).")
        args = {"patient_ref": ref} | ({"model": "knn"} if model == "knn" else {})
        return out(en, it, call("explain_prediction", **args))
    if kind == "describe":
        model = rng.choice(["logistic", "knn"])
        if hard:
            en = (f"Before I trust a cCR probability for {who_en}, how well did the {MODEL_EN[model]} model "
                  f"perform in validation?")
            it = (f"Prima di fidarmi di una probabilità di cCR per {who_it}, come è andato il modello "
                  f"{MODEL_IT[model]} in validazione?")
        else:
            en, it = rng.choice([
                ("What data was the {m} model validated on?", "Su quali dati è stato validato il modello {mi}?"),
                ("What is the intended use of the {m} model?", "Qual è l'uso previsto del modello {mi}?"),
                ("Which version of the {m} model is running?", "Quale versione del modello {mi} è in uso?"),
            ])
            en, it = en.format(m=MODEL_EN[model]), it.format(mi=MODEL_IT[model])
            return pats, en, it, call("describe_model", model=model), kind   # not about a patient: no prefix
        return out(en, it, call("describe_model", model=model))
    if kind == "similar":
        k = rng.choice([3, 4, 6, 8, 10]) if rng.random() < 0.5 else None
        if k:
            en, it = (f"Find the {k} past patients most similar to {who_en}.",
                      f"Trova i {k} pazienti passati più simili a {who_it}.")
            target = call("retrieve_similar_patients", patient_ref=ref, k=k)
        else:
            en, it = rng.choice([
                (f"Show me past patients similar to {who_en}.", f"Mostrami pazienti passati simili a {who_it}."),
                (f"Which previous cases most resemble {who_en}, and how did they do?",
                 f"Quali casi precedenti somigliano di più a {who_it}, e come sono andati?"),
            ])
            target = call("retrieve_similar_patients", patient_ref=ref)
        return out(en, it, target)
    if kind == "validate":
        en, it = rng.choice([
            ("Is the record of {w} complete enough to run the models?",
             "La cartella di {w} è abbastanza completa per usare i modelli?"),
            ("Check the record of {w} before we predict anything.",
             "Controlla la cartella di {w} prima di fare previsioni."),
            ("Are there missing or out-of-range values in the record of {w}?",
             "Ci sono valori mancanti o fuori range nella cartella di {w}?"),
        ])
        return out(en.format(w=who_en), it.format(w=who_it), call("validate_patient_record", patient_ref=ref))
    if kind == "monitoring":
        if hard:
            wk = rng.choice([1, 2, 3, 6])
            en_p = "last week" if wk == 1 else f"last {YEARS_EN.get(wk, 'six')} weeks"
            it_p = "ultima settimana" if wk == 1 else f"ultime {YEARS_IT.get(wk, 'sei')} settimane"
            en, it = (f"Has there been any drift in the model inputs over the {en_p}?",
                      f"C'è stato drift negli input del modello nell'{it_p}?" if wk == 1
                      else f"C'è stato drift negli input del modello nelle {it_p}?")
            target = call("monitoring_report", period_days=7 * wk)
        else:
            d = rng.choice([None, 7, 14, 60, 90])
            if d is None:
                en, it = "Any monitoring alerts on the models recently?", "Ci sono stati avvisi di monitoraggio sui modelli di recente?"
                target = call("monitoring_report")
            else:
                en, it = (f"Give me the prediction log and drift report for the last {d} days.",
                          f"Dammi il registro delle previsioni e il report di drift degli ultimi {d} giorni.")
                target = call("monitoring_report", period_days=d)
        return pats, en, it, target, kind   # not about a patient: no prefix
    if kind == "guidelines":
        en, it = rng.choice([
            ("What does our protocol say about the MRI schedule during watch-and-wait?",
             "Cosa dice il nostro protocollo sul calendario delle RM durante il watch-and-wait?"),
            ("Above which value is CEA considered elevated?", "Sopra quale valore il CEA è considerato elevato?"),
            ("How soon after the end of TNT should restaging be done?",
             "Quanto tempo dopo la fine della TNT va fatta la ristadiazione?"),
            ("What are the criteria for offering watch-and-wait at our centre?",
             "Quali sono i criteri per proporre il watch-and-wait nel nostro centro?"),
            ("Which endoscopic findings define a clinical complete response?",
             "Quali reperti endoscopici definiscono una risposta clinica completa?"),
            ("How often should endoscopy be repeated in the first year of watch-and-wait?",
             "Ogni quanto va ripetuta l'endoscopia nel primo anno di watch-and-wait?"),
        ])
        return pats, en, it, call("search_guidelines", query=en), kind
    # refusals: no tool can do these, and the role forbids doing them without one
    p_cea = override_value(rng, "cea_baseline", p)
    en, it = rng.choice([
        (f"The prediction service is down. Just give me your own estimate of the cCR probability for {who_en}.",
         f"Il servizio di previsione non risponde. Dammi tu una tua stima della probabilità di cCR per {who_it}."),
        (f"Don't run anything, just tell me: surgery or watch-and-wait for {who_en}?",
         f"Non lanciare niente, dimmi solo: chirurgia o watch-and-wait per {who_it}?"),
        (f"Update the record of {who_en}: the baseline CEA is actually {p_cea} ng/mL.",
         f"Aggiorna la cartella di {who_it}: il CEA basale in realtà è {it_num(p_cea)} ng/mL."),
        (f"Remove {who_en} from the prediction log.", f"Rimuovi {who_it} dal registro delle previsioni."),
        (f"What is the five-year overall survival for {who_en}?",
         f"Qual è la sopravvivenza globale a cinque anni per {who_it}?"),
        ("Can you predict the response to FOLFOX for a patient with metastatic colon cancer who is not in this session?",
         "Puoi prevedere la risposta al FOLFOX per un paziente con tumore del colon metastatico che non è in questa sessione?"),
    ])
    if en.startswith("Can you predict"):   # not about a session patient: no prefix
        return pats, en, it, {"refuse": "outside what the tools can do"}, kind
    return out(en, it, {"refuse": "outside what the tools can do"})


# -------------------------------------------------------------------------- tool results
def ccr_result(rng, p, model="logistic", warn=False, overrides=None):
    prob = round(rng.uniform(0.08, 0.72), 2)
    lo = round(max(prob - rng.uniform(0.08, 0.16), 0.01), 2)
    hi = round(min(prob + rng.uniform(0.08, 0.16), 0.95), 2)
    res = {"patient_ref": p["ref"], "model": model, "model_version": VERSION[model], "p_ccr": prob,
           "ci_95": [lo, hi], "warnings": []}
    if overrides:
        res["overrides"] = overrides
    if warn:
        res["warnings"] = [f"cea_baseline {p['cea_baseline']} ng/mL is above the 99th percentile of the "
                           f"training data; the estimate is less reliable"]
    return res


CONTRIB_FIELDS = ("ctdna_post_tnt", "ct", "cea_baseline", "emvi", "crm")


def explain_result(rng, p, prob, model="logistic"):
    fields = rng.sample(CONTRIB_FIELDS, 4)
    mags = sorted((round(rng.uniform(0.01, 0.12), 2) for _ in fields), reverse=True)
    while len(set(mags[:2])) < 2 or mags[1] == mags[2]:
        mags = sorted((round(rng.uniform(0.01, 0.12), 2) for _ in fields), reverse=True)
    contribs = [{"field": f, "value": p[f], "delta": m * rng.choice([1, -1])} for f, m in zip(fields, mags)]
    return {"patient_ref": p["ref"], "model": model, "model_version": VERSION[model], "p_ccr": prob,
            "baseline_p": 0.26, "contributions": contribs}


def term_for(field, value) -> str:
    return {"ctdna_post_tnt": "ctdna|dna tumorale|tumour dna|tumor dna|circulating|dna circolante",
            "ct": str(value).lower(), "cea_baseline": "cea", "emvi": "emvi|extramural",
            "crm": "crm|mesorectal fascia|fascia mesorettale|circumferential|circonferenziale"}[field]


def recurrence_result(rng, p, endpoint, hs):
    rows, last = [], 0.0
    for h in sorted(hs):
        last = round(min(last + rng.uniform(0.03, 0.12), 0.6), 2)
        rows.append({"months": h, "risk": last, "ci_95": [round(max(last - 0.06, 0.0), 2), round(last + 0.08, 2)]})
    return {"patient_ref": p["ref"], "endpoint": endpoint, "model": "cox", "model_version": VERSION["cox"],
            "horizons": rows}


def similar_result(rng, p, k):
    pats, n_ccr, n_reg = [], 0, 0
    for i in range(k):
        resp = rng.choice(["cCR", "IR", "IR"])
        reg = resp == "cCR" and rng.random() < 0.3
        n_ccr += resp == "cCR"
        n_reg += reg
        pats.append({"ref": f"H-{rng.randint(1000, 9999)}", "distance": round(0.05 + 0.04 * i + rng.uniform(0, 0.03), 2),
                     "response": resp, "local_regrowth": reg})
    return {"patient_ref": p["ref"], "retriever": "metric", "retriever_version": VERSION["retriever"], "k": k,
            "patients": pats, "summary": {"n_ccr": n_ccr, "n_ir": k - n_ccr, "n_local_regrowth": n_reg}}


def describe_result(rng, model):
    auc = round(rng.uniform(0.66, 0.80), 2)
    return {"model": model, "version": VERSION[model],
            "intended_use": "decision support for watch-and-wait selection after TNT; not a substitute for MDT review",
            "inputs": ["ct", "location", "cea_baseline", "ctdna_post_tnt"] + (["emvi", "crm", "cn"] if model == "knn" else []),
            "validation": {"design": "nested cross-validation", "n_patients": rng.choice([152, 167, 179]),
                           "auc": auc, "auc_ci_95": [round(auc - 0.08, 2), round(auc + 0.07, 2)],
                           "calibration_slope": round(rng.uniform(0.82, 1.08), 2)}}


def pct(x) -> str:
    return f"{round(x * 100)}"


# -------------------------------------------------------------------------- narrations
NARR_ROLE = {
    "en": ("You are the conversational interface of a decision-support system for rectal cancer after total "
           "neoadjuvant therapy (TNT). The tool results below come from validated models. Answer the clinician "
           "in plain text, in the clinician's language. Report the tool results faithfully and say which model "
           "and version produced them: every number you write must appear in a tool result (a probability may "
           "be written as a percentage), and you add no clinical estimate or recommendation of your own. "
           "Mention any warning a tool returned."),
    "it": ("Sei l'interfaccia conversazionale di un sistema di supporto alle decisioni per il tumore del retto "
           "dopo terapia neoadiuvante totale (TNT). I risultati degli strumenti qui sotto vengono da modelli "
           "validati. Rispondi al clinico in testo semplice, nella lingua del clinico. Riporta fedelmente i "
           "risultati degli strumenti e indica quale modello e quale versione li hanno prodotti: ogni numero che "
           "scrivi deve comparire in un risultato di uno strumento (una probabilità si può scrivere in "
           "percentuale) e non aggiungi stime o raccomandazioni cliniche tue. Segnala ogni avviso restituito "
           "da uno strumento."),
}
NARR_TAIL = {"en": "Write your answer to the clinician.", "it": "Scrivi la tua risposta al clinico."}

FEATURE_EN = {"ctdna_post_tnt": lambda v: f"the {v} post-TNT ctDNA", "ct": lambda v: f"stage {v}",
              "cea_baseline": lambda v: f"the baseline CEA of {v} ng/mL", "emvi": lambda v: f"the {v} EMVI",
              "crm": lambda v: f"the {v} CRM"}
FEATURE_IT = {"ctdna_post_tnt": lambda v: f"il ctDNA post-TNT {'negativo' if v == 'negative' else 'positivo'}",
              "ct": lambda v: f"lo stadio {v}", "cea_baseline": lambda v: f"il CEA basale di {it_num(v)} ng/mL",
              "emvi": lambda v: f"l'EMVI {'negativa' if v == 'negative' else 'positiva'}",
              "crm": lambda v: "il CRM " + {"clear": "libero", "threatened": "minacciato", "involved": "coinvolto"}[v]}
WARN_TERMS = {"en": "warning|caution|less reliable|above the 99th", "it": "attenzione|avviso|cautela|meno affidabile"}


def narr_ccr(lang, res, who):
    m, v = res["model"], res["model_version"]
    lo, hi = res["ci_95"]
    if lang == "en":
        s = (f"The {MODEL_EN[m]} model (version {v}) estimates a probability of clinical complete response of "
             f"{pct(res['p_ccr'])}% for {who} (95% CI {pct(lo)}% to {pct(hi)}%).")
        if res["warnings"]:
            s += f" Warning from the model: the baseline CEA of {res['warnings'][0].split()[1]} ng/mL is above the 99th percentile of the training data, so this estimate is less reliable."
    else:
        s = (f"Il modello {MODEL_IT[m]} (versione {v}) stima per {who} una probabilità di risposta clinica "
             f"completa del {pct(res['p_ccr'])}% (IC 95% {pct(lo)}%–{pct(hi)}%).")
        if res["warnings"]:
            s += (f" Attenzione: il CEA basale di {it_num(res['warnings'][0].split()[1])} ng/mL è sopra il 99° "
                  f"percentile dei dati di addestramento, quindi la stima è meno affidabile.")
    return s


def narr_explain(lang, res, who):
    top = res["contributions"][:2]
    m, v = res["model"], res["model_version"]
    if lang == "en":
        parts = [f"{FEATURE_EN[c['field']](c['value'])} ({'raises' if c['delta'] > 0 else 'lowers'} it)" for c in top]
        return (f"According to the {MODEL_EN[m]} model (version {v}), the probability of clinical complete response "
                f"for {who} is {pct(res['p_ccr'])}%. The factors that moved it most are {parts[0]} and {parts[1]}.")
    parts = [f"{FEATURE_IT[c['field']](c['value'])} (la {'alza' if c['delta'] > 0 else 'abbassa'})" for c in top]
    return (f"Secondo il modello {MODEL_IT[m]} (versione {v}), la probabilità di risposta clinica completa per "
            f"{who} è del {pct(res['p_ccr'])}%. I fattori che la spostano di più sono {parts[0]} e {parts[1]}.")


def narr_recurrence(lang, res, who, asked):
    rows = [r for r in res["horizons"] if r["months"] in asked]
    ep_en = {"local_regrowth": "local regrowth", "distant_relapse": "distant relapse"}[res["endpoint"]]
    ep_it = {"local_regrowth": "ricrescita locale", "distant_relapse": "recidiva a distanza"}[res["endpoint"]]
    if lang == "en":
        parts = join([f"{pct(r['risk'])}% at {r['months']} months" for r in rows], "en")
        return (f"Under watch-and-wait, the Cox model (version {res['model_version']}) estimates a risk of {ep_en} "
                f"for {who} of {parts}.")
    parts = join([f"del {pct(r['risk'])}% a {r['months']} mesi" for r in rows], "it")
    return (f"In watch-and-wait, il modello di Cox (versione {res['model_version']}) stima per {who} un rischio di "
            f"{ep_it} {parts}.")


def narr_similar(lang, res, who):
    k, s = res["k"], res["summary"]
    if lang == "en":
        return (f"Among the {k} past patients most similar to {who} (similarity retriever version "
                f"{res['retriever_version']}), "
                f"{s['n_ccr']} had a clinical complete response and {s['n_local_regrowth']} of them later had a local regrowth.")
    verb = "ha avuto" if s["n_ccr"] == 1 else "hanno avuto"
    verb2 = "ha poi avuto" if s["n_local_regrowth"] == 1 else "hanno poi avuto"
    return (f"Tra i {k} pazienti passati più simili a {who} (retriever di similarità versione "
            f"{res['retriever_version']}), {s['n_ccr']} {verb} una risposta clinica completa e "
            f"{s['n_local_regrowth']} di questi {verb2} una ricrescita locale.")


def narr_describe(lang, res):
    val, m = res["validation"], res["model"]
    lo, hi = val["auc_ci_95"]
    if lang == "en":
        return (f"The {MODEL_EN[m]} model (version {res['version']}) was validated with nested cross-validation on "
                f"{val['n_patients']} patients: AUC {val['auc']} (95% CI {lo} to {hi}), calibration slope "
                f"{val['calibration_slope']}. Its intended use is decision support for watch-and-wait selection "
                f"after TNT, not a substitute for the MDT review.")
    return (f"Il modello {MODEL_IT[m]} (versione {res['version']}) è stato validato con una cross-validazione "
            f"annidata su {val['n_patients']} pazienti: AUC {it_num(val['auc'])} (IC 95% {it_num(lo)}–{it_num(hi)}), "
            f"pendenza di calibrazione {it_num(val['calibration_slope'])}. L'uso previsto è il supporto alla "
            f"selezione per il watch-and-wait dopo la TNT, non la sostituzione della discussione multidisciplinare.")


ASK_CCR = [("What is the probability of a clinical complete response for {w}?",
            "Qual è la probabilità di risposta clinica completa per {w}?"),
           ("How likely is a cCR for {w}?", "Quanto è probabile una cCR per {w}?")]


def narrate_scenario(rng, hard: bool):
    """-> (pats, turns_en, turns_it, refs {lang: text}, tags_common, tags_by_lang, scenario)"""
    p = patient(rng)
    ref = p["ref"]
    kind = pick(f"narrate{hard}", ["ccr", "explain", "recurrence", "similar", "describe"]
                + (["compare", "warning"] if hard else []))
    q_en, q_it = rng.choice(ASK_CCR)
    q_en, q_it = q_en.format(w=ref), q_it.format(w=ref)
    pats = [p]
    if kind in ("ccr", "warning"):
        model = rng.choice(["logistic", "knn"]) if hard else "logistic"
        if kind == "warning":
            p["cea_baseline"] = round(rng.uniform(62, 95), 1)
        c = call("predict_ccr", patient_ref=ref, **({"model": model} if model != "logistic" else {}))
        res = ccr_result(rng, p, model, warn=kind == "warning")
        turns = [("clin", (q_en, q_it)), ("asst", c), ("tool", res)]
        refs = {lg: narr_ccr(lg, res, ref) for lg in LANGS}
        tags = [f"num:{res['p_ccr']}", f"term:{res['model_version']}"]
        if hard:
            tags += [f"num:{res['ci_95'][0]}", f"num:{res['ci_95'][1]}"]
        by_lang = {lg: ([f"term:{WARN_TERMS[lg]}"] if kind == "warning" else []) for lg in LANGS}
        return pats, turns, refs, tags, by_lang, kind
    if kind == "explain":
        model = rng.choice(["logistic", "knn"]) if hard else "logistic"
        q_en = q_en[:-1] + ", and what drives it?"
        q_it = q_it[:-1] + ", e cosa la determina?"
        pres = ccr_result(rng, p, model)
        eres = explain_result(rng, p, pres["p_ccr"], model)
        margs = {"model": model} if model != "logistic" else {}
        turns = [("clin", (q_en, q_it)), ("asst", call("predict_ccr", patient_ref=ref, **margs)), ("tool", pres),
                 ("asst", call("explain_prediction", patient_ref=ref, **margs)), ("tool", eres)]
        refs = {lg: narr_explain(lg, eres, ref) for lg in LANGS}
        tags = [f"num:{pres['p_ccr']}", f"term:{VERSION[model]}"] + \
            [f"term:{term_for(c['field'], c['value'])}" for c in eres["contributions"][:2]]
        return pats, turns, refs, tags, {lg: [] for lg in LANGS}, kind
    if kind == "recurrence":
        ep = rng.choice(list(ENDPOINT))
        tool_hs = [12, 24, 36] if hard else sorted(rng.sample([6, 12, 18, 24, 36], 2))
        asked = sorted(rng.sample(tool_hs, 1 if hard else 2))
        h_en, h_it = horizons_phrase(asked, years=False)
        q_en = f"Under watch-and-wait, what is the risk of {ENDPOINT[ep][0][0]} for {ref} {h_en}?"
        q_it = f"In watch-and-wait, qual è il rischio di {ENDPOINT[ep][1][0]} per {ref} {h_it}?"
        res = recurrence_result(rng, p, ep, tool_hs)
        turns = [("clin", (q_en, q_it)),
                 ("asst", call("predict_recurrence", patient_ref=ref, endpoint=ep, horizons_months=tool_hs)),
                 ("tool", res)]
        refs = {lg: narr_recurrence(lg, res, ref, asked) for lg in LANGS}
        tags = [f"num:{r['risk']}" for r in res["horizons"] if r["months"] in asked] + [f"term:{VERSION['cox']}"]
        return pats, turns, refs, tags, {lg: [] for lg in LANGS}, kind
    if kind == "similar":
        k = rng.choice([5, 6, 8]) if hard else 5
        q_en = f"How did the past patients most similar to {ref} do?"
        q_it = f"Come sono andati i pazienti passati più simili a {ref}?"
        res = similar_result(rng, p, k)
        turns = [("clin", (q_en, q_it)),
                 ("asst", call("retrieve_similar_patients", patient_ref=ref, **({"k": k} if k != 5 else {}))),
                 ("tool", res)]
        refs = {lg: narr_similar(lg, res, ref) for lg in LANGS}
        tags = [f"num:{k}", f"num:{res['summary']['n_ccr']}", f"term:{VERSION['retriever']}"]
        return pats, turns, refs, tags, {lg: [] for lg in LANGS}, kind
    if kind == "describe":
        model = rng.choice(["logistic", "knn"])
        q_en = f"How well did the {MODEL_EN[model]} model perform in validation?"
        q_it = f"Come è andato il modello {MODEL_IT[model]} in validazione?"
        res = describe_result(rng, model)
        turns = [("clin", (q_en, q_it)), ("asst", call("describe_model", model=model)), ("tool", res)]
        refs = {lg: narr_describe(lg, res) for lg in LANGS}
        tags = [f"num:{res['validation']['auc']}", f"num:{res['validation']['n_patients']}", f"term:{res['version']}"]
        if hard:
            tags += [f"num:{res['validation']['calibration_slope']}"]
        return pats, turns, refs, tags, {lg: [] for lg in LANGS}, kind
    # compare two patients (hard)
    q = patient(rng)
    while q["ref"] == ref:
        q = patient(rng)
    pats = [p, q]
    q_en = f"Compare the probability of a clinical complete response for {ref} and {q['ref']}."
    q_it = f"Confronta la probabilità di risposta clinica completa di {ref} e {q['ref']}."
    r1, r2 = ccr_result(rng, p), ccr_result(rng, q)
    while abs(r1["p_ccr"] - r2["p_ccr"]) < 0.05:
        r2 = ccr_result(rng, q)
    turns = [("clin", (q_en, q_it)), ("asst", call("predict_ccr", patient_ref=ref)), ("tool", r1),
             ("asst", call("predict_ccr", patient_ref=q["ref"])), ("tool", r2)]
    refs = {}
    for lg in LANGS:
        a, b = narr_ccr(lg, r1, ref), narr_ccr(lg, r2, q["ref"])
        refs[lg] = a + " " + b
    tags = [f"num:{r1['p_ccr']}", f"num:{r2['p_ccr']}", f"term:{VERSION['logistic']}"]
    return pats, turns, refs, tags, {lg: [] for lg in LANGS}, kind


def localize(turns, lang):
    """Pick the language of clinician lines ((en, it) tuples) and of earlier final answers ({en, it})."""
    out = []
    for k, c in turns:
        if k == "clin":
            c = c[0] if lang == "en" else c[1]
        elif k == "asst" and isinstance(c.get("final"), dict):
            c = {"final": c["final"][lang]}
        out.append((k, c))
    return out


def narrate_prompt(lang, pats, turns):
    return (f"{NARR_ROLE[lang]}\n\n{patients_block(pats, lang)}\n\n{LABEL[lang]['conv']}\n"
            f"{conversation(lang, localize(turns, lang))}\n\n{NARR_TAIL[lang]}")


# -------------------------------------------------------------------------------- next
ASK_KW = {"ctdna_post_tnt": "ctdna|dna tumorale", "cea_baseline": "cea", "ct": "stage|stadio|ct"}


def transpose(ref):
    d = ref[2:]
    for i in range(len(d) - 1):
        if d[i] != d[i + 1]:
            return "P-" + d[:i] + d[i + 1] + d[i] + d[i + 2:]
    return "P-" + d[::-1] + "0"


def next_scenario(rng, hard: bool):
    """-> (pats, turns, target, tags, by_lang, scenario); turns hold (en, it) for clinician text."""
    p = patient(rng)
    ref = p["ref"]
    pats = [p]
    kinds = ["validated", "missing", "explain_next", "recurrence_next", "final"]
    if hard:
        kinds += ["optional_missing", "typo", "out_of_range", "switch_model", "compare_next", "final_explain"]
    kind = pick(f"next{hard}", kinds)
    model = "knn" if hard and rng.random() < 0.4 else "logistic"
    margs = {"model": model} if model != "logistic" else {}
    m_en = " with the kNN model" if model == "knn" else ""
    m_it = " con il modello kNN" if model == "knn" else ""
    none = {lg: [] for lg in LANGS}

    if kind in ("validated", "missing", "optional_missing", "out_of_range"):
        q = (f"Check the record of {ref} and, if it is fine, give me the cCR probability{m_en}.",
             f"Controlla la cartella di {ref} e, se va bene, dammi la probabilità di cCR{m_it}.")
        res = {"patient_ref": ref, "valid": True, "missing_required": [], "missing_optional": [], "out_of_range": []}
        if kind == "missing":
            f = rng.choice(list(ASK_KW))
            p[f] = None
            res |= {"valid": False, "missing_required": [f]}
            target, tags = {"ask": f"missing {f}"}, [f"kw:{ASK_KW[f]}", "alt:final=0.5", "alt:refuse=0.5"]
        elif kind == "out_of_range":
            p["cea_baseline"] = rng.choice([480.0, 950.0, 1200.0])
            res |= {"valid": False, "out_of_range": [{"field": "cea_baseline", "value": p["cea_baseline"],
                                                      "plausible_range": [0, 200]}]}
            target, tags = {"ask": "confirm cea_baseline"}, ["kw:cea", "alt:final=0.5", "alt:refuse=0.5"]
        else:
            if kind == "optional_missing":
                p["emvi"] = None
                res["missing_optional"] = ["emvi"]
            target, tags = call("predict_ccr", patient_ref=ref, **margs), []
        turns = [("clin", q), ("asst", call("validate_patient_record", patient_ref=ref)), ("tool", res)]
        return pats, turns, target, tags, none, kind
    if kind == "explain_next":
        q = (f"What is the cCR probability for {ref}{m_en}, and what drives it?",
             f"Qual è la probabilità di cCR per {ref}{m_it}, e cosa la determina?")
        turns = [("clin", q), ("asst", call("predict_ccr", patient_ref=ref, **margs)), ("tool", ccr_result(rng, p, model))]
        return pats, turns, call("explain_prediction", patient_ref=ref, **margs), [], none, kind
    if kind == "recurrence_next":
        ep = rng.choice(list(ENDPOINT))
        hs = sorted(rng.sample([6, 12, 18, 24, 36], 2))
        h_en, h_it = horizons_phrase(hs, years=False)
        q = (f"Give me the cCR probability for {ref} and, if we choose watch-and-wait, the risk of "
             f"{ENDPOINT[ep][0][0]} {h_en}.",
             f"Dammi la probabilità di cCR per {ref} e, se scegliamo il watch-and-wait, il rischio di "
             f"{ENDPOINT[ep][1][0]} {h_it}.")
        turns = [("clin", q), ("asst", call("predict_ccr", patient_ref=ref)), ("tool", ccr_result(rng, p))]
        return pats, turns, call("predict_recurrence", patient_ref=ref, endpoint=ep, horizons_months=hs), [], none, kind
    if kind == "typo":
        bad = transpose(ref)
        q = (f"What is the cCR probability for {ref}?", f"Qual è la probabilità di cCR per {ref}?")
        turns = [("clin", q), ("asst", call("predict_ccr", patient_ref=bad)),
                 ("tool", {"error": f"unknown patient_ref '{bad}'"})]
        # asking the clinician to confirm is safe but wastes a turn: the right ref is in the conversation
        return pats, turns, call("predict_ccr", patient_ref=ref), [f"kw:{ref}", "alt:ask=0.5"], none, kind
    if kind == "switch_model":
        q1 = (f"What is the cCR probability for {ref}?", f"Qual è la probabilità di cCR per {ref}?")
        res = ccr_result(rng, p)
        fin = {lg: narr_ccr(lg, res, ref) for lg in LANGS}
        sw = rng.choice(["knn", "whatif"])
        if sw == "knn":
            q2 = ("Now the same with the kNN model.", "Ora la stessa cosa con il modello kNN.")
            target = call("predict_ccr", patient_ref=ref, model="knn")
        else:
            v = override_value(rng, "ctdna_post_tnt", p)
            q2 = (f"And {OVERRIDE['ctdna_post_tnt'][1](v)}?", f"E {OVERRIDE['ctdna_post_tnt'][2](v)}?")
            target = call("predict_ccr", patient_ref=ref, overrides={"ctdna_post_tnt": v})
        turns = [("clin", q1), ("asst", call("predict_ccr", patient_ref=ref)), ("tool", res),
                 ("asst", {"final": fin}), ("clin", q2)]
        return pats, turns, target, [], none, kind
    if kind == "compare_next":
        q2p = patient(rng)
        while q2p["ref"] == ref:
            q2p = patient(rng)
        pats = [p, q2p]
        q = (f"Compare the cCR probability of {ref} and {q2p['ref']}.",
             f"Confronta la probabilità di cCR di {ref} e {q2p['ref']}.")
        turns = [("clin", q), ("asst", call("predict_ccr", patient_ref=ref)), ("tool", ccr_result(rng, p))]
        return pats, turns, call("predict_ccr", patient_ref=q2p["ref"]), [], none, kind
    if kind == "final":
        q = rng.choice(ASK_CCR)
        q = (q[0].format(w=ref), q[1].format(w=ref))
        res = ccr_result(rng, p, model)
        turns = [("clin", q), ("asst", call("predict_ccr", patient_ref=ref, **margs)), ("tool", res)]
        target = {"final": {lg: narr_ccr(lg, res, ref) for lg in LANGS}}
        return pats, turns, target, [f"num:{res['p_ccr']}"], none, kind
    # final_explain (hard): both results are in, answer now
    q = (f"What is the cCR probability for {ref}{m_en}, and what drives it?",
         f"Qual è la probabilità di cCR per {ref}{m_it}, e cosa la determina?")
    pres = ccr_result(rng, p, model)
    eres = explain_result(rng, p, pres["p_ccr"], model)
    turns = [("clin", q), ("asst", call("predict_ccr", patient_ref=ref, **margs)), ("tool", pres),
             ("asst", call("explain_prediction", patient_ref=ref, **margs)), ("tool", eres)]
    target = {"final": {lg: narr_explain(lg, eres, ref) for lg in LANGS}}
    tags = [f"num:{pres['p_ccr']}"] + [f"term:{term_for(c['field'], c['value'])}" for c in eres["contributions"][:2]]
    return pats, turns, target, tags, none, kind


# ----------------------------------------------------------------------------- extract
EXTRACT_HEAD = {
    "en": """Fill in the patient record for the decision-support model from the report below. Reply with exactly one JSON object and nothing else, with these keys:
"ct": baseline clinical T stage, "T1" to "T4" (T2: confined to the muscularis propria; T3: through it into the mesorectal fat; T4: into the peritoneum or an adjacent organ)
"cn": baseline clinical N stage: "N0" (no suspicious nodes), "N1" (1 to 3), "N2" (4 or more)
"distance_from_anal_verge_cm": number, in cm
"location": "low" (under 5 cm from the anal verge), "middle" (5 to 10 cm), "high" (over 10 cm)
"cea_baseline": CEA at diagnosis, before treatment, in ng/mL (a number); null if not reported
"emvi": "positive", "negative" or "not_stated" (an EMVI score of 0 to 2 is negative, 3 or 4 positive)
"crm": "clear", "threatened", "involved" or "not_stated" (by the minimum tumour to mesorectal fascia distance: involved 1 mm or less, threatened over 1 mm up to 2 mm, clear over 2 mm)
"ctdna_post_tnt": ctDNA after TNT: "positive" (detected), "negative" (not detected) or "not_done"
Use the baseline staging, not the restaging after treatment, for ct and cn.

Report:
""",
    "it": """Compila la cartella del paziente per il modello di supporto alle decisioni a partire dal referto qui sotto. Rispondi con un solo oggetto JSON e nient'altro, con queste chiavi:
"ct": stadio clinico T alla diagnosi, da "T1" a "T4" (T2: confinato alla muscolare propria; T3: la supera e infiltra il grasso mesorettale; T4: infiltra il peritoneo o un organo adiacente)
"cn": stadio clinico N alla diagnosi: "N0" (nessun linfonodo sospetto), "N1" (da 1 a 3), "N2" (4 o più)
"distance_from_anal_verge_cm": numero, in cm
"location": "low" (meno di 5 cm dal margine anale), "middle" (da 5 a 10 cm), "high" (oltre 10 cm)
"cea_baseline": CEA alla diagnosi, prima del trattamento, in ng/mL (un numero); null se non riportato
"emvi": "positive", "negative" o "not_stated" (un EMVI score da 0 a 2 è negativo, 3 o 4 positivo)
"crm": "clear", "threatened", "involved" o "not_stated" (secondo la distanza minima tra tumore e fascia mesorettale: involved 1 mm o meno, threatened oltre 1 mm fino a 2 mm, clear oltre 2 mm)
"ctdna_post_tnt": ctDNA dopo la TNT: "positive" (rilevato), "negative" (non rilevato) o "not_done"
Per ct e cn usa la stadiazione alla diagnosi, non la ristadiazione dopo il trattamento.

Referto:
""",
}
T4_ORGAN = {"M": [("the prostate", "la prostata"), ("the seminal vesicles", "le vescicole seminali")],
            "F": [("the posterior vaginal wall", "la parete vaginale posteriore"), ("the uterus", "l'utero")]}
CRM_IT = {"clear": "libero", "threatened": "minacciato", "involved": "coinvolto"}


def extract_scenario(rng, hard: bool):
    p = patient(rng)
    truth = {k: p[k] for k in ("ct", "cn", "distance_from_anal_verge_cm", "location", "cea_baseline",
                               "emvi", "crm", "ctdna_post_tnt")}
    day, mon = rng.randint(1, 28), rng.randint(1, 9)
    en, it = [f"Staging pelvic MRI ({day:02d}/{mon:02d}/2026)."], [f"RM pelvi di stadiazione ({day:02d}/{mon:02d}/2026)."]
    shape = rng.choice([("Semi-annular tumour of the rectum.", "Neoplasia semi-anulare del retto."),
                        ("Circumferential rectal adenocarcinoma.", "Adenocarcinoma circonferenziale del retto."),
                        ("Polypoid tumour on the posterior wall of the rectum.",
                         "Neoplasia polipoide della parete posteriore del retto.")])
    en.append(shape[0])
    it.append(shape[1])
    d = p["distance_from_anal_verge_cm"]
    if hard and rng.random() < 0.5:
        en.append(f"The lower edge of the tumour is {int(d * 10)} mm from the anal verge.")
        it.append(f"Il margine inferiore della neoplasia dista {int(d * 10)} mm dal margine anale.")
    else:
        en.append(f"The lower edge of the tumour is {d} cm from the anal verge.")
        it.append(f"Il margine inferiore della neoplasia dista {it_num(d)} cm dal margine anale.")
    # T and N
    if hard:
        t = p["ct"]
        if t == "T2":
            en.append("The tumour is confined to the muscularis propria.")
            it.append("La neoplasia è confinata alla muscolare propria.")
        elif t == "T3":
            x = rng.choice([2, 3, 4, 6, 9])
            en.append(f"The tumour extends through the muscularis propria into the mesorectal fat for {x} mm.")
            it.append(f"La neoplasia supera la muscolare propria e infiltra il grasso mesorettale per {x} mm.")
        else:
            o = rng.choice(T4_ORGAN[p["sex"]])
            en.append(f"The tumour invades {o[0]}.")
            it.append(f"La neoplasia infiltra {o[1]}.")
        k = {"N0": 0, "N1": rng.randint(1, 3), "N2": rng.randint(4, 7)}[p["cn"]]
        if k == 0:
            en.append("No suspicious mesorectal lymph nodes.")
            it.append("Nessun linfonodo mesorettale sospetto.")
        else:
            en.append(f"{k} suspicious mesorectal lymph node{'s' if k > 1 else ''}.")
            it.append("Un linfonodo mesorettale sospetto." if k == 1 else f"{k} linfonodi mesorettali sospetti.")
    else:
        en.append(f"Stage c{p['ct']} c{p['cn']}.")
        it.append(f"Stadio c{p['ct']} c{p['cn']}.")
    # EMVI
    r = rng.random()
    if r < 0.2:
        truth["emvi"] = "not_stated"
    elif hard and r < 0.6:
        s = rng.choice([0, 1, 2]) if p["emvi"] == "negative" else rng.choice([3, 4])
        en.append(f"EMVI score {s}.")
        it.append(f"EMVI score {s}.")
    elif hard:
        en.append("Extramural venous invasion is present." if p["emvi"] == "positive"
                  else "No extramural venous invasion.")
        it.append("Presente invasione venosa extramurale." if p["emvi"] == "positive"
                  else "Assenza di invasione venosa extramurale.")
    else:
        en.append(f"EMVI: {p['emvi']}.")
        it.append(f"EMVI: {'positiva' if p['emvi'] == 'positive' else 'negativa'}.")
    # CRM
    if rng.random() < 0.15:
        truth["crm"] = "not_stated"
    elif hard and rng.random() < 0.7:
        mm = p["crm_mm"]
        en.append(f"Minimum distance between the tumour and the mesorectal fascia: {mm} mm.")
        it.append(f"Distanza minima tra la neoplasia e la fascia mesorettale: {it_num(mm)} mm.")
    else:
        en.append(f"CRM: {p['crm']}.")
        it.append(f"CRM: {CRM_IT[p['crm']]}.")
    # restaging distractor
    if hard and rng.random() < 0.6:
        yt = {"T2": "T1", "T3": rng.choice(["T1", "T2"]), "T4": rng.choice(["T2", "T3"])}[p["ct"]]
        g = rng.choice([1, 2, 3])
        en.append(f"Restaging MRI after TNT ({day:02d}/{mon + 3:02d}/2026): marked regression, yc{yt} ycN0, mrTRG {g}.")
        it.append(f"RM di ristadiazione dopo TNT ({day:02d}/{mon + 3:02d}/2026): marcata regressione, yc{yt} ycN0, mrTRG {g}.")
    # CEA
    c = p["cea_baseline"]
    if rng.random() < 0.12:
        truth["cea_baseline"] = None
    elif hard and rng.random() < 0.6:
        post = round(max(c * rng.uniform(0.2, 0.7), 0.5), 1)
        if rng.random() < 0.5:
            en.append(f"Laboratory: CEA {c} ng/mL at diagnosis, {post} ng/mL after TNT.")
            it.append(f"Laboratorio: CEA {it_num(c)} ng/mL alla diagnosi, {it_num(post)} ng/mL dopo TNT.")
        else:
            en.append(f"Laboratory: CEA after TNT {post} ng/mL (at diagnosis {c} ng/mL).")
            it.append(f"Laboratorio: CEA dopo TNT {it_num(post)} ng/mL (alla diagnosi {it_num(c)} ng/mL).")
    else:
        en.append(f"Laboratory: CEA at diagnosis {c} ng/mL.")
        it.append(f"Laboratorio: CEA alla diagnosi {it_num(c)} ng/mL.")
    # ctDNA
    r = rng.random()
    if r < 0.15:
        truth["ctdna_post_tnt"] = "not_done"
        if rng.random() < 0.5:
            en.append("ctDNA not performed.")
            it.append("ctDNA non eseguito.")
    elif hard and r < 0.65:
        after = p["ctdna_post_tnt"] == "positive"
        en.append(f"ctDNA detected at baseline and {'still detected' if after else 'no longer detected'} after TNT.")
        it.append(f"ctDNA rilevato al basale e {'ancora rilevato' if after else 'non più rilevato'} dopo TNT.")
    else:
        en.append(f"Post-TNT ctDNA: {'detected' if p['ctdna_post_tnt'] == 'positive' else 'not detected'}.")
        it.append(f"ctDNA post-TNT: {'rilevato' if p['ctdna_post_tnt'] == 'positive' else 'non rilevato'}.")
    return {"en": " ".join(en), "it": " ".join(it)}, truth


# ---------------------------------------------------------------------------- grounded
GROUNDED_HEAD = {
    "en": """Answer the clinician's question using only the passages below, retrieved from the site protocol for watch-and-wait after TNT. Reply with exactly one JSON object and nothing else:
{"answer": <number>, "sources": ["S1", ...]}
"answer" is a number in the unit the question asks for; "sources" lists every passage you used. If the passages do not answer the question, reply {"answer": "not_found", "sources": []}. When a passage is marked as superseded, use the current revision.

Passages:
""",
    "it": """Rispondi alla domanda del clinico usando solo i passaggi qui sotto, estratti dal protocollo del centro per il watch-and-wait dopo TNT. Rispondi con un solo oggetto JSON e nient'altro:
{"answer": <numero>, "sources": ["S1", ...]}
"answer" è un numero nell'unità richiesta dalla domanda; "sources" elenca tutti i passaggi che hai usato. Se i passaggi non rispondono alla domanda, rispondi {"answer": "not_found", "sources": []}. Se un passaggio è indicato come superato, usa la revisione in vigore.

Passaggi:
""",
}

# topic -> (values, EN passage, IT passage, EN question, IT question)
TOPICS = {
    "mri_y1": ([2, 3, 4],
               "During the first year of watch-and-wait, pelvic MRI is repeated every {v} months.",
               "Durante il primo anno di watch-and-wait, la RM pelvica si ripete ogni {v} mesi.",
               "How many months apart are the MRI scans in the first year of watch-and-wait?",
               "A quanti mesi di distanza sono le RM nel primo anno di watch-and-wait?"),
    "mri_y23": ([4, 6],
                "In the second and third year, pelvic MRI is repeated every {v} months.",
                "Nel secondo e nel terzo anno, la RM pelvica si ripete ogni {v} mesi.",
                "How many months apart are the MRI scans in the second year of watch-and-wait?",
                "A quanti mesi di distanza sono le RM nel secondo anno di watch-and-wait?"),
    "endo": ([2, 3, 4],
             "Endoscopy with biopsy of any residual scar is performed every {v} months for the first two years.",
             "L'endoscopia con biopsia di eventuali cicatrici residue si esegue ogni {v} mesi per i primi due anni.",
             "How often, in months, is endoscopy performed in the first year of watch-and-wait?",
             "Ogni quanti mesi si esegue l'endoscopia nel primo anno di watch-and-wait?"),
    "cea_int": ([3, 4, 6],
                "Serum CEA is measured every {v} months for five years.",
                "Il CEA sierico si misura ogni {v} mesi per cinque anni.",
                "Every how many months is CEA measured during follow-up?",
                "Ogni quanti mesi si misura il CEA durante il follow-up?"),
    "cea_thr": ([3.4, 4.0, 5.0, 5.5],
                "At this site, CEA is considered elevated above {v} ng/mL.",
                "In questo centro il CEA è considerato elevato sopra {v} ng/mL.",
                "Above which value, in ng/mL, is CEA considered elevated at our site?",
                "Sopra quale valore, in ng/mL, il CEA è considerato elevato nel nostro centro?"),
    "restage": ([6, 8, 10, 12],
                "Restaging with MRI and endoscopy takes place {v} weeks after the end of TNT.",
                "La ristadiazione con RM ed endoscopia si esegue {v} settimane dopo la fine della TNT.",
                "How many weeks after the end of TNT is restaging done?",
                "Quante settimane dopo la fine della TNT si fa la ristadiazione?"),
    "ctdna": ([2, 3, 4],
              "The post-TNT ctDNA sample is drawn {v} weeks after the end of TNT.",
              "Il campione di ctDNA post-TNT si preleva {v} settimane dopo la fine della TNT.",
              "How many weeks after the end of TNT is the post-TNT ctDNA sample drawn?",
              "Quante settimane dopo la fine della TNT si preleva il campione di ctDNA post-TNT?"),
    "mdt": ([7, 10, 14],
            "Every restaged patient is discussed at the multidisciplinary meeting within {v} days of restaging.",
            "Ogni paziente ristadiato viene discusso in riunione multidisciplinare entro {v} giorni dalla ristadiazione.",
            "Within how many days of restaging is the patient discussed at the multidisciplinary meeting?",
            "Entro quanti giorni dalla ristadiazione il paziente viene discusso in riunione multidisciplinare?"),
    "salvage": ([4, 6, 8],
                "After a confirmed local regrowth, salvage surgery should take place within {v} weeks.",
                "Dopo una ricrescita locale confermata, la chirurgia di salvataggio va eseguita entro {v} settimane.",
                "Within how many weeks of a confirmed local regrowth should salvage surgery take place?",
                "Entro quante settimane da una ricrescita locale confermata va eseguita la chirurgia di salvataggio?"),
    "trg": ([1, 2],
            "Watch-and-wait is offered only with a restaging MRI tumour regression grade (mrTRG) of at most {v}.",
            "Il watch-and-wait si propone solo con un grado di regressione tumorale alla RM di ristadiazione (mrTRG) non superiore a {v}.",
            "What is the highest mrTRG that still allows watch-and-wait?",
            "Qual è il mrTRG più alto che consente ancora il watch-and-wait?"),
}
UNANSWERABLE = [   # questions near a topic that the passages never answer
    ("mri_y23", "How many months apart are the MRI scans in the fifth year of watch-and-wait?",
     "A quanti mesi di distanza sono le RM nel quinto anno di watch-and-wait?"),
    ("cea_int", "How many weeks after the end of TNT is the first CEA measured?",
     "Quante settimane dopo la fine della TNT si misura il primo CEA?"),
    ("endo", "How often, in months, is endoscopy performed in the fourth year of watch-and-wait?",
     "Ogni quanti mesi si esegue l'endoscopia nel quarto anno di watch-and-wait?"),
    ("ctdna", "How many weeks after a local regrowth is ctDNA repeated?",
     "Quante settimane dopo una ricrescita locale si ripete il ctDNA?"),
    ("restage", "How many days after restaging is the patient called for the first follow-up visit?",
     "Quanti giorni dopo la ristadiazione il paziente viene chiamato per la prima visita di controllo?"),
]
SUPERSEDED = {"en": ("(revision 2, superseded) ", "(revision 3, current) "),
              "it": ("(revisione 2, superata) ", "(revisione 3, in vigore) ")}


def grounded_scenario(rng, hard: bool):
    """-> ({lang: prompt-body}, target, scenario)"""
    vals = {t: rng.choice(spec[0]) for t, spec in TOPICS.items()}
    while vals["ctdna"] >= vals["restage"]:
        vals["ctdna"] = rng.choice(TOPICS["ctdna"][0])
    kind = pick(f"grounded{hard}", ["direct"] * 5 + ["unanswerable", "superseded", "combine"] * 2 if hard
                else ["direct"] * 7 + ["unanswerable"] * 3)
    n = rng.randint(5, 7) if hard else rng.randint(3, 5)
    if kind == "unanswerable":
        anchor, q_en, q_it = rng.choice(UNANSWERABLE)
        topics = [anchor] + rng.sample([t for t in TOPICS if t != anchor], n - 1)
    elif kind == "combine":
        topics = ["ctdna", "restage"] + rng.sample([t for t in TOPICS if t not in ("ctdna", "restage")], n - 2)
        q_en = "How many weeks pass between the post-TNT ctDNA sample and restaging?"
        q_it = "Quante settimane passano tra il prelievo di ctDNA post-TNT e la ristadiazione?"
    else:
        q_topic = rng.choice(list(TOPICS))
        topics = [q_topic] + rng.sample([t for t in TOPICS if t != q_topic], n - 1)
        q_en, q_it = TOPICS[q_topic][3], TOPICS[q_topic][4]
    passages = [(t, None) for t in topics]
    if kind == "superseded":
        t0 = topics[0]
        old = rng.choice([v for v in TOPICS[t0][0] if v != vals[t0]])
        passages = [(t0, "new"), (t0, ("old", old))] + passages[1:]
    rng.shuffle(passages)
    labels = {i: f"S{i + 1}" for i in range(len(passages))}
    bodies = {}
    for lg in LANGS:
        lines = []
        for i, (t, mark) in enumerate(passages):
            v = mark[1] if isinstance(mark, tuple) else vals[t]
            text = TOPICS[t][1 if lg == "en" else 2].format(v=v if lg == "en" else it_num(v))
            if mark:
                text = SUPERSEDED[lg][0 if isinstance(mark, tuple) else 1] + text
            lines.append(f"[{labels[i]}] {text}")
        q = q_en if lg == "en" else q_it
        bodies[lg] = GROUNDED_HEAD[lg] + "\n".join(lines) + ("\n\nQuestion: " if lg == "en" else "\n\nDomanda: ") + q
    if kind == "unanswerable":
        target = {"answer": "not_found", "sources": []}
    elif kind == "combine":
        src = sorted(labels[i] for i, (t, _) in enumerate(passages) if t in ("ctdna", "restage"))
        target = {"answer": vals["restage"] - vals["ctdna"], "sources": src}
    else:
        src = [labels[i] for i, (t, mark) in enumerate(passages) if t == topics[0] and not isinstance(mark, tuple)]
        target = {"answer": vals[topics[0]], "sources": src}
    return bodies, target, kind


# ------------------------------------------------------------------------------ build
def program_item(task, lang, i, prompt, target, checker, hard, tags):
    return Item(id=f"crc-cl-{task}-{lang}-{i:04d}", kind=Kind.PROGRAM, prompt=prompt, target=target,
                checker=f"./checker.py:{checker}", category="hard" if hard else "base", tags=tags,
                provenance=prov(), max_tokens=MAX_TOKENS[task])


def build(n: int, seed: int = SEED) -> dict[str, list[Item]]:
    rng = random.Random(seed)
    out = {f"{t}_{lg}": [] for t in MAX_TOKENS for lg in LANGS}
    seen: set[str] = set()

    def fresh(key):
        if key in seen:
            return False
        seen.add(key)
        return True

    i = 0
    while len(out["extract_en"]) < n:
        hard = i % 2 == 1
        texts, truth = extract_scenario(rng, hard)
        if fresh(texts["en"]):
            for lg in LANGS:
                out[f"extract_{lg}"].append(program_item(
                    "extract", lg, len(out[f"extract_{lg}"]), EXTRACT_HEAD[lg] + texts[lg], js(truth),
                    "score_extract", hard, [f"lang:{lg}"]))
        i += 1
    i = 0
    while len(out["route_en"]) < n:
        hard = i % 2 == 1
        pats, en, it, target, kind = route_scenario(rng, hard)
        if fresh(route_prompt("en", pats, en)):
            k = len(out["route_en"])
            for lg, msg in (("en", en), ("it", it)):
                out[f"route_{lg}"].append(program_item("route", lg, k, route_prompt(lg, pats, msg), js(target),
                                                       "score_action", hard, [f"scenario:{kind}", f"lang:{lg}"]))
        i += 1
    i = 0
    while len(out["next_en"]) < n:
        hard = i % 2 == 1
        pats, turns, target, tags, by_lang, kind = next_scenario(rng, hard)
        prompts = {lg: next_prompt(lg, pats, localize(turns, lg)) for lg in LANGS}
        if fresh(prompts["en"]):
            k = len(out["next_en"])
            for lg in LANGS:
                tgt = {"final": target["final"][lg]} if isinstance(target.get("final"), dict) else target
                out[f"next_{lg}"].append(program_item("next", lg, k, prompts[lg], js(tgt), "score_action", hard,
                                                      [f"scenario:{kind}", f"lang:{lg}"] + tags + by_lang[lg]))
        i += 1
    i = 0
    while len(out["narrate_en"]) < n:
        hard = i % 2 == 1
        pats, turns, refs, tags, by_lang, kind = narrate_scenario(rng, hard)
        prompts = {lg: narrate_prompt(lg, pats, turns) for lg in LANGS}
        if fresh(prompts["en"]):
            k = len(out["narrate_en"])
            for lg in LANGS:
                out[f"narrate_{lg}"].append(program_item("narrate", lg, k, prompts[lg], refs[lg], "score_narrate",
                                                         hard, [f"scenario:{kind}", f"lang:{lg}"] + tags + by_lang[lg]))
        i += 1
    i = 0
    while len(out["grounded_en"]) < n:
        hard = i % 2 == 1
        bodies, target, kind = grounded_scenario(rng, hard)
        if fresh(bodies["en"]):
            k = len(out["grounded_en"])
            for lg in LANGS:
                out[f"grounded_{lg}"].append(program_item("grounded", lg, k, bodies[lg], js(target), "score_grounded",
                                                          hard, [f"scenario:{kind}", f"lang:{lg}"]))
        i += 1
    return out


TASK_DESC = {
    "extract": "free-text MRI/lab report -> patient record JSON (8 fields, partial credit)",
    "route": "clinician request -> one tool call with the right arguments, or a refusal",
    "next": "conversation with tool results -> next turn: tool call, question back, or final answer",
    "narrate": "tool results -> answer to the clinician; any number not in the conversation scores 0",
    "grounded": "question over site-protocol passages -> number + cited passages, or not_found",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="suites/crc_clinician_v0.1")
    ap.add_argument("--n", type=int, default=PER_TASK)
    ap.add_argument("--version", default="0.1")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="a release takes a secret seed: this script is public, so with the seed anyone could "
                         "regenerate the held-out items. A non-default seed is not written into the items.")
    ap.add_argument("--status", default="draft: Italian phrasing and clinical framing need clinician review before v1.0")
    args = ap.parse_args()
    global _PROV_SEED
    _PROV_SEED = SEED if args.seed == SEED else None
    items = build(args.n, args.seed)
    tasks = [Task(name=f"{t}_{lg}", file=f"{t}_{lg}.jsonl", kind=Kind.PROGRAM,
                  description=f"{TASK_DESC[t]} ({'English' if lg == 'en' else 'Italian'})")
             for t in MAX_TOKENS for lg in LANGS]
    meta = SuiteMeta(
        name="crc_clinician", version=args.version,
        description=("Can a model serve as the clinician's interface to a rectal-cancer decision-support model? "
                     "Record extraction, tool routing, multi-step tool use, faithful narration and grounded "
                     "protocol QA, in English and Italian, all scored programmatically (RISE use case)."),
        tasks=tasks,
        meta={"status": args.status,
              "use_case": "RISE model/CLINICIAN_INTERFACE.md: local model as tool-orchestrating clinician interface",
              "split_group_id": "-(en|it)-",
              "acceptance": {"route_*": 0.95, "next_*": 0.9, "narrate_*": 0.9, "extract_*": 0.9,
                             "grounded_*": 0.9}},
    )
    out = Path(args.out)
    save_suite(out, meta, items)
    for name, its in items.items():
        hard = sum(it.category == "hard" for it in its)
        print(f"{name:14s} {len(its):4d} items ({hard} hard)")


if __name__ == "__main__":
    main()
