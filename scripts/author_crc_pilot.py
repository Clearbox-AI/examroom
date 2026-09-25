"""Author the CRC pilot suite: SLM capability ladder for colorectal-cancer drug-efficacy work.

Ten tasks, five tiers × two languages (EN/IT), all scored programmatically:

  tier 1  mcq_*      choice_generate  domain knowledge (hand-written, NEEDS EXPERT REVIEW)
  tier 2  extract_*  numeric          find one number in a synthetic trial abstract
  tier 3  compute_*  numeric          derive a number (risk difference, NNT, pooled ORR)
  tier 4  summary_*  program          abstract -> fixed-schema JSON (./checker.py:score_summary)
  tier 5  tool_*     program          emit an execute_python tool call (./checker.py:score_tool_call)

Tiers 2-5 are template-generated with seeded jitter (no leakage, expandable);
all numbers use the decimal POINT in both languages and prompts say so, because
`first_number` treats "34,5" as 345 (thousands separator).

Since v0.3 the efficacy direction of every synthetic abstract is a coin flip: the
experimental arm (always the first regimen mentioned) wins half the time, and the
toxicity and subgroup directions are independent coins, so neither "the first
regimen" nor "the larger number" is a shortcut. Items carry a `direction:exp|ctrl`
tag so results can be split by it.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from benchsmith.items import Item, Kind, Provenance, SuiteMeta, Task, save_suite

SEED = 7000
PROV = lambda: Provenance(method="template", source="scripts/author_crc_pilot.py", seed=SEED)
HAND = lambda: Provenance(method="hand", source="scripts/author_crc_pilot.py", author="luca@clearbox.ai")
DIR = lambda v: [f"direction:{'exp' if v['exp_wins'] else 'ctrl'}"]  # which arm the abstract favours

# ---------------------------------------------------------------------------- tier 1: MCQ bank
# (q_en, q_it, opts_en, opts_it) — correct option FIRST; positions are rebalanced below.
# Draft status: verify every fact with a domain expert before freezing a v1.
MCQ = [
    ("Before starting cetuximab or panitumumab in metastatic colorectal cancer, which molecular result is required?",
     "Prima di iniziare cetuximab o panitumumab nel carcinoma colorettale metastatico, quale risultato molecolare è richiesto?",
     ["KRAS and NRAS wild-type", "BRAF V600E mutation", "high microsatellite instability", "HER2 amplification"],
     ["KRAS e NRAS wild-type", "mutazione BRAF V600E", "instabilità dei microsatelliti elevata", "amplificazione di HER2"]),
    ("What is the mechanism of action of bevacizumab?",
     "Qual è il meccanismo d'azione del bevacizumab?",
     ["it binds circulating VEGF-A", "it blocks the EGFR extracellular domain", "it inhibits topoisomerase I", "it blocks PD-1 on T cells"],
     ["lega il VEGF-A circolante", "blocca il dominio extracellulare di EGFR", "inibisce la topoisomerasi I", "blocca PD-1 sui linfociti T"]),
    ("Which first-line treatment is preferred for MSI-high / mismatch-repair-deficient metastatic colorectal cancer?",
     "Quale trattamento di prima linea è preferito per il carcinoma colorettale metastatico MSI-high / con deficit del mismatch repair?",
     ["pembrolizumab", "FOLFOX plus bevacizumab", "regorafenib", "trifluridine-tipiracil"],
     ["pembrolizumab", "FOLFOX più bevacizumab", "regorafenib", "trifluridina-tipiracil"]),
    ("Which combination is approved for previously treated BRAF V600E-mutant metastatic colorectal cancer?",
     "Quale combinazione è approvata per il carcinoma colorettale metastatico con mutazione BRAF V600E già trattato?",
     ["encorafenib plus cetuximab", "vemurafenib alone", "dabrafenib plus trametinib", "FOLFOX plus panitumumab"],
     ["encorafenib più cetuximab", "vemurafenib in monoterapia", "dabrafenib più trametinib", "FOLFOX più panitumumab"]),
    ("A deficiency of which enzyme causes severe toxicity from 5-fluorouracil and capecitabine?",
     "Il deficit di quale enzima causa tossicità grave da 5-fluorouracile e capecitabina?",
     ["dihydropyrimidine dehydrogenase (DPD)", "UDP-glucuronosyltransferase 1A1", "thiopurine methyltransferase", "CYP2D6"],
     ["diidropirimidina deidrogenasi (DPD)", "UDP-glucuronosiltransferasi 1A1", "tiopurina metiltransferasi", "CYP2D6"]),
    ("Which germline variant increases the risk of severe neutropenia and diarrhea from irinotecan?",
     "Quale variante germinale aumenta il rischio di neutropenia grave e diarrea da irinotecan?",
     ["UGT1A1*28", "DPYD*2A", "TPMT*3A", "CYP2C19*2"],
     ["UGT1A1*28", "DPYD*2A", "TPMT*3A", "CYP2C19*2"]),
    ("What is the characteristic dose-limiting toxicity of oxaliplatin?",
     "Qual è la tossicità dose-limitante caratteristica dell'oxaliplatino?",
     ["cumulative peripheral sensory neuropathy", "dilated cardiomyopathy", "hemorrhagic cystitis", "pulmonary fibrosis"],
     ["neuropatia sensitiva periferica cumulativa", "cardiomiopatia dilatativa", "cistite emorragica", "fibrosi polmonare"]),
    ("Capecitabine is an oral prodrug of which agent?",
     "La capecitabina è un profarmaco orale di quale agente?",
     ["5-fluorouracil", "irinotecan", "oxaliplatin", "methotrexate"],
     ["5-fluorouracile", "irinotecan", "oxaliplatino", "metotrexato"]),
    ("Which drugs make up the FOLFIRI regimen?",
     "Quali farmaci compongono il regime FOLFIRI?",
     ["folinic acid, 5-fluorouracil and irinotecan", "folinic acid, 5-fluorouracil and oxaliplatin", "capecitabine and oxaliplatin", "5-fluorouracil, oxaliplatin and irinotecan"],
     ["acido folinico, 5-fluorouracile e irinotecan", "acido folinico, 5-fluorouracile e oxaliplatino", "capecitabina e oxaliplatino", "5-fluorouracile, oxaliplatino e irinotecan"]),
    ("To which drug class does regorafenib belong?",
     "A quale classe di farmaci appartiene il regorafenib?",
     ["oral multikinase inhibitor", "anti-EGFR monoclonal antibody", "PD-L1 inhibitor", "antifolate"],
     ["inibitore multichinasico orale", "anticorpo monoclonale anti-EGFR", "inibitore di PD-L1", "antifolato"]),
    ("Trifluridine-tipiracil (TAS-102) combines a thymidine analogue with an inhibitor of which enzyme?",
     "La trifluridina-tipiracil (TAS-102) combina un analogo della timidina con un inibitore di quale enzima?",
     ["thymidine phosphorylase", "thymidylate synthase", "dihydrofolate reductase", "topoisomerase II"],
     ["timidina fosforilasi", "timidilato sintasi", "diidrofolato reduttasi", "topoisomerasi II"]),
    ("Anti-EGFR antibodies give the greatest benefit in RAS wild-type tumours arising in which location?",
     "Gli anticorpi anti-EGFR danno il massimo beneficio nei tumori RAS wild-type che originano in quale sede?",
     ["left colon and rectum", "right colon", "caecum only", "appendix"],
     ["colon sinistro e retto", "colon destro", "solo cieco", "appendice"]),
    ("Immunohistochemistry for mismatch-repair deficiency tests which four proteins?",
     "L'immunoistochimica per il deficit del mismatch repair valuta quali quattro proteine?",
     ["MLH1, MSH2, MSH6 and PMS2", "BRCA1, BRCA2, ATM and PALB2", "KRAS, NRAS, BRAF and PIK3CA", "p53, RB1, APC and SMAD4"],
     ["MLH1, MSH2, MSH6 e PMS2", "BRCA1, BRCA2, ATM e PALB2", "KRAS, NRAS, BRAF e PIK3CA", "p53, RB1, APC e SMAD4"]),
    ("Which combination targets KRAS G12C-mutant metastatic colorectal cancer after prior therapy?",
     "Quale combinazione ha come bersaglio il carcinoma colorettale metastatico con mutazione KRAS G12C dopo terapia precedente?",
     ["sotorasib plus panitumumab", "osimertinib plus cetuximab", "alectinib plus bevacizumab", "imatinib plus 5-fluorouracil"],
     ["sotorasib più panitumumab", "osimertinib più cetuximab", "alectinib più bevacizumab", "imatinib più 5-fluorouracile"]),
    ("Which regimen is an option for HER2-amplified, RAS wild-type metastatic colorectal cancer?",
     "Quale regime è un'opzione per il carcinoma colorettale metastatico HER2-amplificato e RAS wild-type?",
     ["tucatinib plus trastuzumab", "tamoxifen plus letrozole", "rituximab plus bendamustine", "erlotinib plus gemcitabine"],
     ["tucatinib più trastuzumab", "tamoxifene più letrozolo", "rituximab più bendamustina", "erlotinib più gemcitabina"]),
    ("For low-risk stage III colon cancer treated with adjuvant CAPOX, the IDEA collaboration supports what duration?",
     "Per il carcinoma del colon in stadio III a basso rischio trattato con CAPOX adiuvante, la collaborazione IDEA supporta quale durata?",
     ["3 months", "6 months", "12 months", "18 months"],
     ["3 mesi", "6 mesi", "12 mesi", "18 mesi"]),
    ("In microsatellite-stable metastatic colorectal cancer, single-agent PD-1 blockade is generally…",
     "Nel carcinoma colorettale metastatico con microsatelliti stabili, il blocco di PD-1 in monoterapia è generalmente…",
     ["ineffective", "curative", "the preferred first line", "active only after radiotherapy"],
     ["inefficace", "curativo", "la prima linea preferita", "attivo solo dopo radioterapia"]),
    ("Aflibercept, added to FOLFIRI in second line, acts as a decoy receptor for which ligands?",
     "L'aflibercept, aggiunto a FOLFIRI in seconda linea, agisce come recettore esca per quali ligandi?",
     ["VEGF-A, VEGF-B and placental growth factor", "EGF and TGF-alpha", "HER2 and HER3", "interleukin-6"],
     ["VEGF-A, VEGF-B e fattore di crescita placentare", "EGF e TGF-alfa", "HER2 e HER3", "interleuchina-6"]),
    ("In refractory metastatic colorectal cancer, adding which drug to trifluridine-tipiracil improved overall survival in the SUNLIGHT trial?",
     "Nel carcinoma colorettale metastatico refrattario, l'aggiunta di quale farmaco alla trifluridina-tipiracil ha migliorato la sopravvivenza globale nello studio SUNLIGHT?",
     ["bevacizumab", "cetuximab", "pembrolizumab", "oxaliplatin"],
     ["bevacizumab", "cetuximab", "pembrolizumab", "oxaliplatino"]),
    ("5-fluorouracil exerts its main cytotoxic effect by inhibiting which enzyme?",
     "Il 5-fluorouracile esercita il suo principale effetto citotossico inibendo quale enzima?",
     ["thymidylate synthase", "ribonucleotide reductase", "DNA polymerase alpha", "aromatase"],
     ["timidilato sintasi", "ribonucleotide reduttasi", "DNA polimerasi alfa", "aromatasi"]),
    ("Why is folinic acid (leucovorin) given together with 5-fluorouracil?",
     "Perché l'acido folinico (leucovorin) viene somministrato insieme al 5-fluorouracile?",
     ["it stabilises the inhibitory complex on thymidylate synthase", "it prevents peripheral neuropathy", "it acts as an antiemetic", "it protects the kidneys"],
     ["stabilizza il complesso inibitorio sulla timidilato sintasi", "previene la neuropatia periferica", "agisce da antiemetico", "protegge i reni"]),
    ("What is the mechanism of action of irinotecan?",
     "Qual è il meccanismo d'azione dell'irinotecan?",
     ["topoisomerase I inhibition", "microtubule stabilisation", "proteasome inhibition", "PARP inhibition"],
     ["inibizione della topoisomerasi I", "stabilizzazione dei microtubuli", "inibizione del proteasoma", "inibizione di PARP"]),
    ("Neoadjuvant dostarlimab produced complete clinical responses in locally advanced rectal cancers with which feature?",
     "Il dostarlimab neoadiuvante ha prodotto risposte cliniche complete nei carcinomi del retto localmente avanzati con quale caratteristica?",
     ["mismatch-repair deficiency", "HER2 amplification", "KRAS G12D mutation", "microsatellite stability"],
     ["deficit del mismatch repair", "amplificazione di HER2", "mutazione KRAS G12D", "stabilità dei microsatelliti"]),
    ("In colorectal cancer care, serum CEA is mainly used for…",
     "Nel carcinoma colorettale, il CEA sierico è usato principalmente per…",
     ["monitoring for recurrence and treatment response", "population screening", "choosing between FOLFOX and FOLFIRI", "diagnosing Lynch syndrome"],
     ["monitorare recidive e risposta al trattamento", "screening di popolazione", "scegliere tra FOLFOX e FOLFIRI", "diagnosticare la sindrome di Lynch"]),
]


def build_mcq() -> tuple[list[Item], list[Item]]:
    rng = random.Random(SEED)
    en, it = [], []
    for i, (q_en, q_it, o_en, o_it) in enumerate(MCQ):
        pos = i % 4  # balanced correct-answer positions
        order = [0] + rng.sample([1, 2, 3], 3)
        order.remove(0)
        order.insert(pos, 0)
        for lang, q, opts, out in (("en", q_en, o_en, en), ("it", q_it, o_it, it)):
            out.append(Item(id=f"crc-mcq-{lang}-{i:03d}", kind=Kind.CHOICE_GENERATE, prompt=q,
                            options=[opts[j] for j in order], answer_idx=pos,
                            category=lang, tags=[f"pair:{i:03d}"], provenance=HAND()))
    return en, it


# ---------------------------------------------------------------- tiers 2-4: synthetic abstracts
REGIMENS = ["FOLFOX{j}bevacizumab", "FOLFIRI{j}cetuximab", "CAPOX", "FOLFOXIRI{j}bevacizumab",
            "FOLFIRI{j}panitumumab", "FOLFOX", "FOLFIRI{j}aflibercept", "CAPOX{j}bevacizumab", "FOLFIRI"]
POPS = [("RAS wild-type", "RAS wild-type"), ("KRAS-mutant", "con mutazione KRAS"),
        ("BRAF V600E-mutant", "con mutazione BRAF V600E"), ("MSI-high", "MSI-high")]

ABS_EN = ("Background: This randomized phase {ph} trial compared {exp} with {ctrl} in {n} patients "
          "with previously untreated {pop} metastatic colorectal cancer.\n"
          "Results: The objective response rate was {orr_e}% in the {exp} arm versus {orr_c}% in the "
          "{ctrl} arm. Median progression-free survival was {pfs_e} months versus {pfs_c} months "
          "(hazard ratio {hr}; 95% CI {lo} to {hi}). Median overall survival was {os_e} months versus "
          "{os_c} months. Grade 3 or higher adverse events occurred in {ae_e}% versus {ae_c}% of patients.")
ABS_IT = ("Introduzione: questo studio randomizzato di fase {ph} ha confrontato {exp} con {ctrl} in {n} "
          "pazienti con carcinoma colorettale metastatico {pop} non precedentemente trattato.\n"
          "Risultati: il tasso di risposta obiettiva è stato del {orr_e}% nel braccio {exp} contro il "
          "{orr_c}% nel braccio {ctrl}. La sopravvivenza libera da progressione mediana è stata di "
          "{pfs_e} mesi contro {pfs_c} mesi (hazard ratio {hr}; IC 95% da {lo} a {hi}). La sopravvivenza "
          "globale mediana è stata di {os_e} mesi contro {os_c} mesi. Eventi avversi di grado 3 o "
          "superiore si sono verificati nel {ae_e}% contro il {ae_c}% dei pazienti.")


def sample_arms(rng: random.Random) -> tuple[str, str]:
    while True:
        exp, ctrl = rng.sample(REGIMENS, 2)
        # avoid same-backbone pairs (e.g. CAPOX vs CAPOX+bev): the efficacy direction is
        # random, and "adding a drug makes things worse" would read implausibly
        if exp.split("{j}")[0] != ctrl.split("{j}")[0]:
            return exp, ctrl


def make_abstract(rng: random.Random) -> dict:
    while True:
        exp, ctrl = sample_arms(rng)
        # efficacy direction is a coin flip: the experimental arm (always the first regimen
        # mentioned) wins half the time, so neither "first" nor "larger" is a shortcut;
        # toxicity direction is a second, independent coin
        exp_wins, exp_toxic = rng.random() < 0.5, rng.random() < 0.5
        win = dict(orr=round(rng.uniform(35, 68), 1), pfs=round(rng.uniform(8.2, 14.8), 1), os=round(rng.uniform(24.2, 34.8), 1))
        lose = dict(orr=round(rng.uniform(15, 33), 1), pfs=round(rng.uniform(4.1, 7.9), 1), os=round(rng.uniform(14.1, 23.8), 1))
        e, c = (win, lose) if exp_wins else (lose, win)
        hr = round(rng.uniform(0.55, 0.92), 2)
        if not exp_wins:
            hr = round(1 / hr, 2)  # PFS favours control -> hazard ratio above 1
        ae_hi, ae_lo = rng.randrange(45, 72), rng.randrange(24, 44)
        v = dict(ph=rng.choice(["2", "3"]), n=rng.randrange(180, 900, 4),
                 orr_e=e["orr"], orr_c=c["orr"], pfs_e=e["pfs"], pfs_c=c["pfs"], os_e=e["os"], os_c=c["os"],
                 hr=hr, ae_e=ae_hi if exp_toxic else ae_lo, ae_c=ae_lo if exp_toxic else ae_hi, exp_wins=exp_wins)
        v["lo"], v["hi"] = round(hr - rng.uniform(0.06, 0.14), 2), round(hr + rng.uniform(0.06, 0.16), 2)
        nums = [v[k] for k in ("orr_e", "orr_c", "pfs_e", "pfs_c", "os_e", "os_c", "hr", "ae_e", "ae_c", "lo", "hi")]
        if len(set(nums)) == len(nums):  # every quantity unique -> questions are unambiguous
            break
    pop = rng.choice(POPS)
    v["exp_en"], v["ctrl_en"] = exp.format(j=" plus "), ctrl.format(j=" plus ")
    v["exp_it"], v["ctrl_it"] = exp.format(j=" più "), ctrl.format(j=" più ")
    v["pop_en"], v["pop_it"] = pop
    return v


EXTRACT_QS = [  # (field, q_en(v), q_it(v))
    ("orr_e", "What was the objective response rate, in percent, in the {exp} arm?",
              "Qual è stato il tasso di risposta obiettiva, in percentuale, nel braccio {exp}?"),
    ("orr_c", "What was the objective response rate, in percent, in the {ctrl} arm?",
              "Qual è stato il tasso di risposta obiettiva, in percentuale, nel braccio {ctrl}?"),
    ("pfs_e", "What was the median progression-free survival, in months, in the {exp} arm?",
              "Qual è stata la sopravvivenza libera da progressione mediana, in mesi, nel braccio {exp}?"),
    ("hr", "What was the hazard ratio for progression-free survival?",
           "Qual è stato l'hazard ratio per la sopravvivenza libera da progressione?"),
    ("os_c", "What was the median overall survival, in months, in the {ctrl} arm?",
             "Qual è stata la sopravvivenza globale mediana, in mesi, nel braccio {ctrl}?"),
    ("ae_e", "What percentage of patients in the {exp} arm had grade 3 or higher adverse events?",
             "Quale percentuale di pazienti nel braccio {exp} ha avuto eventi avversi di grado 3 o superiore?"),
]
TAIL_EN = " Answer with only the number, using a decimal point."
TAIL_IT = " Rispondi solo con il numero, usando il punto decimale."


def build_extract(n_per_lang=60) -> tuple[list[Item], list[Item]]:
    rng = random.Random(SEED + 1)
    en, it = [], []
    for i in range(n_per_lang):
        v = make_abstract(rng)
        field, q_en, q_it = EXTRACT_QS[i % len(EXTRACT_QS)]
        for lang, tpl, q, tail, out in (("en", ABS_EN, q_en, TAIL_EN, en), ("it", ABS_IT, q_it, TAIL_IT, it)):
            text = tpl.format(exp=v[f"exp_{lang}"], ctrl=v[f"ctrl_{lang}"], pop=v[f"pop_{lang}"], **v)
            question = q.format(exp=v[f"exp_{lang}"], ctrl=v[f"ctrl_{lang}"])
            head = "Question" if lang == "en" else "Domanda"
            out.append(Item(id=f"crc-ext-{lang}-{i:03d}", kind=Kind.NUMERIC,
                            prompt=f"{text}\n\n{head}: {question}{tail}",
                            target=str(v[field]), tolerance=0.001, category=field, tags=DIR(v),
                            max_tokens=48, provenance=PROV()))
    return en, it


COMPUTE_TAIL_EN = (" You may show brief working, then end your reply with a final line in the form "
                   "'ANSWER: <number>', using a decimal point.")
COMPUTE_TAIL_IT = (" Puoi mostrare brevemente i passaggi, poi concludi la risposta con una riga finale "
                   "nella forma 'RISPOSTA: <numero>', usando il punto decimale.")
COMPUTE_QS = {
    "arr": ("What is the absolute difference in response rates between the arms, in percentage points? "
            "Round to one decimal place.",
            "Qual è la differenza assoluta nei tassi di risposta tra i due bracci, in punti percentuali? "
            "Arrotonda a un decimale."),
    "nnt": ("How many patients would need to be treated with {exp} instead of {ctrl} for one additional "
            "response? Round to the nearest whole number.",
            "Quanti pazienti dovrebbero essere trattati con {exp} invece di {ctrl} per ottenere una "
            "risposta aggiuntiva? Arrotonda al numero intero più vicino."),
    "pooled": ("Considering both arms together, what percentage of all patients responded? Round to one decimal place.",
               "Considerando entrambi i bracci insieme, quale percentuale di tutti i pazienti ha risposto? "
               "Arrotonda a un decimale."),
}
STEM_EN = ("In a randomized trial, {a} of {na} patients in the {exp} arm had an objective response, "
           "compared with {b} of {nb} patients in the {ctrl} arm.")
STEM_IT = ("In uno studio randomizzato, {a} pazienti su {na} nel braccio {exp} hanno avuto una risposta "
           "obiettiva, contro {b} pazienti su {nb} nel braccio {ctrl}.")


def build_compute(n_per_lang=24) -> tuple[list[Item], list[Item]]:
    rng = random.Random(SEED + 2)
    kinds = list(COMPUTE_QS)
    en, it = [], []
    for i in range(n_per_lang):
        kind = kinds[i % len(kinds)]
        while True:
            na, nb = rng.randrange(80, 240), rng.randrange(80, 240)
            a, b = rng.randrange(30, na), rng.randrange(10, nb // 2)
            arr = a / na - b / nb
            if arr < 0.10:
                continue
            # 0.15 tolerance accepts answers derived from intermediate one-decimal rounding
            # (e.g. 63.2 - 12.8 = 50.4 when the exact difference rounds to 50.3)
            if kind == "arr":
                target, tol = round(100 * arr, 1), 0.15
            elif kind == "nnt":
                # keep rounding unambiguous: reject 1/ARR within 0.08 of a .5 boundary, where
                # intermediate rounding of the two rates could flip the nearest integer
                # (v0.3 and earlier tested distance to the nearest integer, i.e. the safe cases)
                if abs((1 / arr) % 1 - 0.5) < 0.08:
                    continue
                target, tol = round(1 / arr), 0.5
            else:
                target, tol = round(100 * (a + b) / (na + nb), 1), 0.15
            break
        exp, ctrl = sample_arms(rng)
        v = dict(a=a, b=b, na=na, nb=nb)
        for lang, stem, tail, out in (("en", STEM_EN, COMPUTE_TAIL_EN, en), ("it", STEM_IT, COMPUTE_TAIL_IT, it)):
            e, c = exp.format(j=" plus " if lang == "en" else " più "), ctrl.format(j=" plus " if lang == "en" else " più ")
            q = COMPUTE_QS[kind][0 if lang == "en" else 1].format(exp=e, ctrl=c)
            out.append(Item(id=f"crc-cmp-{lang}-{i:03d}", kind=Kind.NUMERIC,
                            prompt=f"{stem.format(exp=e, ctrl=c, **v)} {q}{tail}",
                            target=str(target), tolerance=tol, category=kind,
                            # v0.1's 320 truncated verbose models mid-derivation (measured
                            # verbosity, not arithmetic); 1200 lets every model finish
                            max_tokens=1200, provenance=PROV()))
    return en, it


SUMMARY_KEYS = ("experimental_arm, control_arm, n_patients, orr_experimental_pct, orr_control_pct, "
                "median_pfs_experimental_months, median_pfs_control_months, hazard_ratio")
SUM_EN = (f"\n\nSummarize this abstract as a JSON object with exactly these keys: {SUMMARY_KEYS}. "
          "The experimental arm is the first regimen mentioned, {exp}. "
          "Use numbers with a decimal point. Reply with only the JSON object.")
SUM_IT = (f"\n\nRiassumi questo abstract come un oggetto JSON con esattamente queste chiavi: {SUMMARY_KEYS}. "
          "Il braccio sperimentale è il primo regime citato, {exp}. "
          "Usa numeri con il punto decimale. Rispondi solo con l'oggetto JSON.")


def build_summary(n_per_lang=40) -> tuple[list[Item], list[Item]]:
    rng = random.Random(SEED + 3)
    en, it = [], []
    for i in range(n_per_lang):
        v = make_abstract(rng)
        for lang, tpl, tail, out in (("en", ABS_EN, SUM_EN, en), ("it", ABS_IT, SUM_IT, it)):
            ref = {"experimental_arm": v[f"exp_{lang}"], "control_arm": v[f"ctrl_{lang}"], "n_patients": v["n"],
                   "orr_experimental_pct": v["orr_e"], "orr_control_pct": v["orr_c"],
                   "median_pfs_experimental_months": v["pfs_e"], "median_pfs_control_months": v["pfs_c"],
                   "hazard_ratio": v["hr"]}
            text = tpl.format(exp=v[f"exp_{lang}"], ctrl=v[f"ctrl_{lang}"], pop=v[f"pop_{lang}"], **v)
            out.append(Item(id=f"crc-sum-{lang}-{i:03d}", kind=Kind.PROGRAM,
                            prompt=text + tail.format(exp=v[f"exp_{lang}"]),
                            target=json.dumps(ref, ensure_ascii=False), checker="./checker.py:score_summary",
                            category=lang, tags=DIR(v), max_tokens=420, provenance=PROV()))
    return en, it


# ------------------------------------------------------------------------- tier 5: tool calls
TOOL_EN = ('You have access to one tool. Call it by replying with a single JSON object and nothing else, '
           'in the form:\n{{"tool": "execute_python", "arguments": {{"code": "<python source>"}}}}\n'
           'A pandas DataFrame named df is already loaded with the clinical trial data below.\n\n'
           '{csv}\n\nTask: using one tool call, {task}')
TOOL_IT = ('Hai a disposizione un solo strumento. Invocalo rispondendo con un unico oggetto JSON e '
           'nient\'altro, nella forma:\n{{"tool": "execute_python", "arguments": {{"code": "<codice python>"}}}}\n'
           'Un DataFrame pandas chiamato df è già caricato con i dati dello studio clinico riportati sotto.\n\n'
           '{csv}\n\nCompito: con una sola chiamata allo strumento, {task}')

TOOL_TASKS = [  # (task_en(v), task_it(v), code(v), requires(v))
    ("print the mean tumor_shrinkage_pct of patients with kras_status equal to '{ks}'",
     "stampa la media di tumor_shrinkage_pct dei pazienti con kras_status uguale a '{ks}'",
     "print(df[df['kras_status'] == '{ks}']['tumor_shrinkage_pct'].mean())",
     ["kras_status", "{ks}", "tumor_shrinkage_pct"]),
    ("print how many patients in the '{arm}' arm have pfs_months greater than {k}",
     "stampa quanti pazienti del braccio '{arm}' hanno pfs_months maggiore di {k}",
     "print(int(((df['arm'] == '{arm}') & (df['pfs_months'] > {k})).sum()))",
     ["arm", "{arm}", "pfs_months", "{k}"]),
    ("print the median pfs_months for each arm",
     "stampa la mediana di pfs_months per ciascun braccio",
     "print(df.groupby('arm')['pfs_months'].median())",
     ["arm", "pfs_months"]),
    ("print the maximum tumor_shrinkage_pct in the '{arm}' arm",
     "stampa il massimo di tumor_shrinkage_pct nel braccio '{arm}'",
     "print(df[df['arm'] == '{arm}']['tumor_shrinkage_pct'].max())",
     ["arm", "{arm}", "tumor_shrinkage_pct"]),
]


def build_tool(n_per_lang=20) -> tuple[list[Item], list[Item]]:
    rng = random.Random(SEED + 4)
    en, it = [], []
    for i in range(n_per_lang):
        arms = rng.sample(["FOLFOX", "FOLFIRI", "CAPOX", "FOLFOXIRI"], 2)
        rows = ["patient_id,arm,kras_status,tumor_shrinkage_pct,pfs_months"]
        for r in range(8):
            rows.append(f"P{r + 1:02d},{arms[r % 2]},{rng.choice(['WT', 'MUT'])},"
                        f"{round(rng.uniform(-20, 80), 1)},{round(rng.uniform(2, 18), 1)}")
        csv = "\n".join(rows)
        sub = dict(ks=rng.choice(["WT", "MUT"]), arm=rng.choice(arms), k=rng.choice([4, 6, 8, 10]))
        t_en, t_it, code, req = TOOL_TASKS[i % len(TOOL_TASKS)]
        code = code.format(**sub)
        target = json.dumps({"tool": "execute_python", "arguments": {"code": code}})
        tags = ["requires:" + r.format(**sub) for r in req]
        for lang, tpl, task, out in (("en", TOOL_EN, t_en, en), ("it", TOOL_IT, t_it, it)):
            out.append(Item(id=f"crc-tool-{lang}-{i:03d}", kind=Kind.PROGRAM,
                            prompt=tpl.format(csv=csv, task=task.format(**sub)),
                            target=target, checker="./checker.py:score_tool_call",
                            category=lang, tags=tags, max_tokens=200, provenance=PROV()))
    return en, it


# ================================================================= HARD BAND (v0.2)
# Restores discrimination at the top end: full-precision 27B-class models ace the
# base band; these require finer knowledge, disambiguation, derivation, and code
# that actually runs.

MCQ_HARD = [
    ("In the BEACON trial, encorafenib plus cetuximab was tested with and without which additional drug?",
     "Nello studio BEACON, encorafenib più cetuximab è stato testato con e senza quale farmaco aggiuntivo?",
     ["binimetinib", "trametinib", "vemurafenib", "panitumumab"],
     ["binimetinib", "trametinib", "vemurafenib", "panitumumab"]),
    ("Which anti-EGFR antibody is a chimeric IgG1, making it more prone to infusion reactions?",
     "Quale anticorpo anti-EGFR è una IgG1 chimerica, più soggetta a reazioni infusionali?",
     ["cetuximab", "panitumumab", "bevacizumab", "ramucirumab"],
     ["cetuximab", "panitumumab", "bevacizumab", "ramucirumab"]),
    ("Severe first-infusion hypersensitivity to cetuximab is associated with pre-existing IgE against which epitope?",
     "L'ipersensibilità grave alla prima infusione di cetuximab è associata a IgE preesistenti contro quale epitopo?",
     ["galactose-alpha-1,3-galactose", "polyethylene glycol", "murine IgG2a", "the beta-lactam ring"],
     ["galattosio-alfa-1,3-galattosio", "polietilenglicole", "IgG2a murina", "l'anello beta-lattamico"]),
    ("The final activation step of capecitabine to 5-FU, enriched in tumour tissue, is catalysed by which enzyme?",
     "L'ultimo passaggio di attivazione della capecitabina a 5-FU, arricchito nel tessuto tumorale, è catalizzato da quale enzima?",
     ["thymidine phosphorylase", "dihydropyrimidine dehydrogenase", "cytidine deaminase", "carboxylesterase"],
     ["timidina fosforilasi", "diidropirimidina deidrogenasi", "citidina deaminasi", "carbossilesterasi"]),
    ("What is the active metabolite of irinotecan?",
     "Qual è il metabolita attivo dell'irinotecan?",
     ["SN-38", "FdUMP", "the oxaliplatin-DACH complex", "N-desmethyl irinotecan"],
     ["SN-38", "FdUMP", "il complesso DACH dell'oxaliplatino", "N-desmetil irinotecan"]),
    ("Which feature distinguishes ACUTE oxaliplatin neurotoxicity from the chronic cumulative form?",
     "Quale caratteristica distingue la neurotossicità ACUTA da oxaliplatino dalla forma cronica cumulativa?",
     ["cold-triggered paraesthesiae appearing within hours to days", "loss of proprioception after months of treatment",
      "permanent distal motor weakness", "predominantly autonomic dysfunction"],
     ["parestesie scatenate dal freddo entro ore o giorni", "perdita della propriocezione dopo mesi di trattamento",
      "deficit motorio distale permanente", "disfunzione prevalentemente autonomica"]),
    ("Regorafenib was approved for refractory metastatic colorectal cancer on the basis of which phase 3 trial?",
     "Il regorafenib è stato approvato per il carcinoma colorettale metastatico refrattario sulla base di quale studio di fase 3?",
     ["CORRECT", "RECOURSE", "SUNLIGHT", "VELOUR"],
     ["CORRECT", "RECOURSE", "SUNLIGHT", "VELOUR"]),
    ("Trifluridine-tipiracil monotherapy was approved on the basis of which phase 3 trial?",
     "La monoterapia con trifluridina-tipiracil è stata approvata sulla base di quale studio di fase 3?",
     ["RECOURSE", "CORRECT", "BEACON", "TRIBE"],
     ["RECOURSE", "CORRECT", "BEACON", "TRIBE"]),
    ("Trastuzumab deruxtecan showed activity in HER2-positive metastatic colorectal cancer in which trial?",
     "Il trastuzumab deruxtecan ha mostrato attività nel carcinoma colorettale metastatico HER2-positivo in quale studio?",
     ["DESTINY-CRC01", "MOUNTAINEER", "KEYNOTE-177", "CheckMate-142"],
     ["DESTINY-CRC01", "MOUNTAINEER", "KEYNOTE-177", "CheckMate-142"]),
    ("Fruquintinib, approved for refractory metastatic colorectal cancer after FRESCO-2, selectively inhibits what?",
     "Il fruquintinib, approvato per il carcinoma colorettale metastatico refrattario dopo FRESCO-2, inibisce selettivamente cosa?",
     ["VEGF receptors 1, 2 and 3", "EGFR and HER2", "BRAF and MEK", "PD-1 and CTLA-4"],
     ["i recettori VEGF 1, 2 e 3", "EGFR e HER2", "BRAF e MEK", "PD-1 e CTLA-4"]),
    ("In anti-EGFR rechallenge strategies such as CHRONOS, patients are selected using which test?",
     "Nelle strategie di rechallenge anti-EGFR come CHRONOS, i pazienti sono selezionati con quale test?",
     ["plasma ctDNA showing RAS/BRAF wild-type status", "repeat tumour biopsy immunohistochemistry",
      "serum CEA doubling time", "FDG-PET metabolic response"],
     ["ctDNA plasmatico con stato RAS/BRAF wild-type", "immunoistochimica su nuova biopsia tumorale",
      "tempo di raddoppiamento del CEA sierico", "risposta metabolica alla FDG-PET"]),
    ("Across colorectal cancer stages, MSI-high frequency is highest in which setting?",
     "Tra gli stadi del carcinoma colorettale, la frequenza di MSI-high è più alta in quale contesto?",
     ["stage II", "stage IV / metastatic disease", "stage III", "it is equal across stages"],
     ["stadio II", "stadio IV / malattia metastatica", "stadio III", "è uguale in tutti gli stadi"]),
    ("Nivolumab plus ipilimumab showed first-line superiority in MSI-high metastatic colorectal cancer in which trial?",
     "Nivolumab più ipilimumab ha mostrato superiorità in prima linea nel carcinoma colorettale metastatico MSI-high in quale studio?",
     ["CheckMate-8HW", "KEYNOTE-177", "CheckMate-227", "BEACON"],
     ["CheckMate-8HW", "KEYNOTE-177", "CheckMate-227", "BEACON"]),
    ("Which adverse-event triad is characteristic of bevacizumab?",
     "Quale triade di eventi avversi è caratteristica del bevacizumab?",
     ["hypertension, proteinuria and gastrointestinal perforation", "neutropenia, alopecia and mucositis",
      "QT prolongation, rash and diarrhoea", "myalgia, oedema and cytopenias"],
     ["ipertensione, proteinuria e perforazione gastrointestinale", "neutropenia, alopecia e mucosite",
      "prolungamento del QT, rash e diarrea", "mialgia, edema e citopenie"]),
    ("5-fluorouracil cardiotoxicity most typically presents as…",
     "La cardiotossicità da 5-fluorouracile si presenta più tipicamente come…",
     ["coronary vasospasm with chest pain", "irreversible dilated cardiomyopathy",
      "complete heart block", "pericardial effusion"],
     ["vasospasmo coronarico con dolore toracico", "cardiomiopatia dilatativa irreversibile",
      "blocco atrioventricolare completo", "versamento pericardico"]),
    ("First-line FOLFOXIRI plus bevacizumab was established by which trial series?",
     "FOLFOXIRI più bevacizumab in prima linea è stato consolidato da quale serie di studi?",
     ["TRIBE", "IDEA", "PRIME", "CRYSTAL"],
     ["TRIBE", "IDEA", "PRIME", "CRYSTAL"]),
]


def build_mcq_hard() -> tuple[list[Item], list[Item]]:
    rng = random.Random(SEED + 10)
    en, it = [], []
    for i, (q_en, q_it, o_en, o_it) in enumerate(MCQ_HARD):
        pos = i % 4
        order = rng.sample([1, 2, 3], 3)
        order.insert(pos, 0)
        for lang, q, opts, out in (("en", q_en, o_en, en), ("it", q_it, o_it, it)):
            out.append(Item(id=f"crc-mcqh-{lang}-{i:03d}", kind=Kind.CHOICE_GENERATE, prompt=q,
                            options=[opts[j] for j in order], answer_idx=pos,
                            category=lang, tags=[f"pair:h{i:03d}"], provenance=HAND()))
    return en, it


SUBGROUPS = [("left-sided primary", "tumore primitivo sinistro"),
             ("liver-limited disease", "malattia limitata al fegato"),
             ("aged 70 or older", "età pari o superiore a 70 anni")]
SUB_EN = " In the {sub} subgroup, the objective response rate was {orr_sub_e}% with {exp} versus {orr_sub_c}% with {ctrl}."
SUB_IT = " Nel sottogruppo con {sub}, il tasso di risposta obiettiva è stato del {orr_sub_e}% con {exp} contro il {orr_sub_c}% con {ctrl}."


def make_hard_abstract(rng: random.Random, taken: set[float]) -> dict:
    while True:
        v = make_abstract(rng)
        sub_hi, sub_lo = round(rng.uniform(20, 75), 1), round(rng.uniform(10, 40), 1)
        sub_exp_wins = rng.random() < 0.5  # subgroup direction: a third independent coin
        v["orr_sub_e"], v["orr_sub_c"] = (sub_hi, sub_lo) if sub_exp_wins else (sub_lo, sub_hi)
        nums = {v[k] for k in ("orr_e", "orr_c", "pfs_e", "pfs_c", "os_e", "os_c",
                               "hr", "ae_e", "ae_c", "lo", "hi", "orr_sub_e", "orr_sub_c")}
        if len(nums) == 13 and not (nums & taken):  # unique within AND across the two studies
            taken |= nums
            return v


EXTRACT_HARD_QS = [
    ("orr_sub_e", "In Study {k}, what was the objective response rate, in percent, in the {exp} arm in the {sub} subgroup?",
                  "Nello Studio {k}, qual è stato il tasso di risposta obiettiva, in percentuale, nel braccio {exp} nel sottogruppo con {sub}?"),
    ("orr_e", "In Study {k}, considering ALL patients, what was the objective response rate, in percent, in the {exp} arm?",
              "Nello Studio {k}, considerando TUTTI i pazienti, qual è stato il tasso di risposta obiettiva, in percentuale, nel braccio {exp}?"),
    ("hr", "What was the hazard ratio for progression-free survival in Study {k}?",
           "Qual è stato l'hazard ratio per la sopravvivenza libera da progressione nello Studio {k}?"),
    ("pfs_c", "In Study {k}, what was the median progression-free survival, in months, in the {ctrl} arm?",
              "Nello Studio {k}, qual è stata la sopravvivenza libera da progressione mediana, in mesi, nel braccio {ctrl}?"),
    ("os_e", "In Study {k}, what was the median overall survival, in months, in the {exp} arm?",
             "Nello Studio {k}, qual è stata la sopravvivenza globale mediana, in mesi, nel braccio {exp}?"),
]


def build_extract_hard(n_per_lang=30) -> tuple[list[Item], list[Item]]:
    rng = random.Random(SEED + 11)
    en, it = [], []
    for i in range(n_per_lang):
        taken: set[float] = set()
        studies = [make_hard_abstract(rng, taken), make_hard_abstract(rng, taken)]
        sub = rng.choice(SUBGROUPS)
        k = rng.choice([1, 2])
        field, q_en, q_it = EXTRACT_HARD_QS[i % len(EXTRACT_HARD_QS)]
        v = studies[k - 1]
        for lang, tpl, subtpl, q, out in (("en", ABS_EN, SUB_EN, q_en, en), ("it", ABS_IT, SUB_IT, q_it, it)):
            blocks = []
            for j, sv in enumerate(studies, 1):
                text = tpl.format(exp=sv[f"exp_{lang}"], ctrl=sv[f"ctrl_{lang}"], pop=sv[f"pop_{lang}"], **sv)
                text += subtpl.format(sub=sub[0 if lang == "en" else 1], exp=sv[f"exp_{lang}"],
                                      ctrl=sv[f"ctrl_{lang}"], **sv)
                blocks.append(("Study" if lang == "en" else "Studio") + f" {j}:\n{text}")
            question = q.format(k=k, exp=v[f"exp_{lang}"], ctrl=v[f"ctrl_{lang}"], sub=sub[0 if lang == "en" else 1])
            head, tail = ("Question", TAIL_EN) if lang == "en" else ("Domanda", TAIL_IT)
            out.append(Item(id=f"crc-exth-{lang}-{i:03d}", kind=Kind.NUMERIC,
                            prompt="\n\n".join(blocks) + f"\n\n{head}: {question}{tail}",
                            target=str(v[field]), tolerance=0.001, category=field, tags=DIR(v),
                            max_tokens=48, provenance=PROV()))
    return en, it


SUMH_KEYS = SUMMARY_KEYS + ", orr_absolute_difference_pct"
SUMH_EN = (f"\n\nSummarize this abstract as a JSON object with exactly these keys: {SUMH_KEYS}. "
           "The experimental arm is the first regimen mentioned, {exp}; orr_absolute_difference_pct is the "
           "experimental minus the control objective response rate, in percentage points. "
           "Use numbers with a decimal point. Reply with only the JSON object.")
SUMH_IT = (f"\n\nRiassumi questo abstract come un oggetto JSON con esattamente queste chiavi: {SUMH_KEYS}. "
           "Il braccio sperimentale è il primo regime citato, {exp}; orr_absolute_difference_pct è il tasso di "
           "risposta obiettiva sperimentale meno quello di controllo, in punti percentuali. "
           "Usa numeri con il punto decimale. Rispondi solo con l'oggetto JSON.")


def build_summary_hard(n_per_lang=24) -> tuple[list[Item], list[Item]]:
    rng = random.Random(SEED + 12)
    en, it = [], []
    for i in range(n_per_lang):
        v = make_abstract(rng)
        for lang, tpl, tail, out in (("en", ABS_EN, SUMH_EN, en), ("it", ABS_IT, SUMH_IT, it)):
            ref = {"experimental_arm": v[f"exp_{lang}"], "control_arm": v[f"ctrl_{lang}"], "n_patients": v["n"],
                   "orr_experimental_pct": v["orr_e"], "orr_control_pct": v["orr_c"],
                   "median_pfs_experimental_months": v["pfs_e"], "median_pfs_control_months": v["pfs_c"],
                   "hazard_ratio": v["hr"],
                   "orr_absolute_difference_pct": round(v["orr_e"] - v["orr_c"], 1)}
            text = tpl.format(exp=v[f"exp_{lang}"], ctrl=v[f"ctrl_{lang}"], pop=v[f"pop_{lang}"], **v)
            out.append(Item(id=f"crc-sumh-{lang}-{i:03d}", kind=Kind.PROGRAM,
                            prompt=text + tail.format(exp=v[f"exp_{lang}"]),
                            target=json.dumps(ref, ensure_ascii=False), checker="./checker.py:score_summary",
                            category=lang, tags=DIR(v), max_tokens=460, provenance=PROV()))
    return en, it


TOOLH_TASKS = [  # (task_en, task_it, code) — expected value computed from the generated rows
    ("print the difference between the mean tumor_shrinkage_pct of the '{a1}' arm and that of the '{a2}' arm, considering only patients with kras_status '{ks}'",
     "stampa la differenza tra la media di tumor_shrinkage_pct del braccio '{a1}' e quella del braccio '{a2}', considerando solo i pazienti con kras_status '{ks}'",
     "sub = df[df['kras_status'] == '{ks}']\n"
     "print(sub[sub['arm'] == '{a1}']['tumor_shrinkage_pct'].mean() - sub[sub['arm'] == '{a2}']['tumor_shrinkage_pct'].mean())"),
    ("print how many patients have pfs_months strictly greater than the mean pfs_months of all patients",
     "stampa quanti pazienti hanno pfs_months strettamente maggiore della media di pfs_months di tutti i pazienti",
     "print(int((df['pfs_months'] > df['pfs_months'].mean()).sum()))"),
    ("print the mean pfs_months of the arm whose mean tumor_shrinkage_pct is higher",
     "stampa la media di pfs_months del braccio con la media di tumor_shrinkage_pct più alta",
     "best = df.groupby('arm')['tumor_shrinkage_pct'].mean().idxmax()\n"
     "print(df[df['arm'] == best]['pfs_months'].mean())"),
    ("print the patient_id of the patient with the highest tumor_shrinkage_pct among those with kras_status '{ks}'",
     "stampa il patient_id del paziente con il tumor_shrinkage_pct più alto tra quelli con kras_status '{ks}'",
     "sub = df[df['kras_status'] == '{ks}']\n"
     "print(sub.loc[sub['tumor_shrinkage_pct'].idxmax(), 'patient_id'])"),
]


def build_tool_hard(n_per_lang=12) -> tuple[list[Item], list[Item]]:
    rng = random.Random(SEED + 13)
    en, it = [], []
    for i in range(n_per_lang):
        kind = i % len(TOOLH_TASKS)
        while True:
            arms = rng.sample(["FOLFOX", "FOLFIRI", "CAPOX", "FOLFOXIRI"], 2)
            rows = [dict(patient_id=f"P{r + 1:02d}", arm=arms[r % 2], kras=rng.choice(["WT", "MUT"]),
                         shrink=round(rng.uniform(-20, 80), 1), pfs=round(rng.uniform(2, 18), 1))
                    for r in range(10)]
            ks = rng.choice(["WT", "MUT"])
            sub = [r for r in rows if r["kras"] == ks]
            # every task must be well-defined on this table
            if len({r["arm"] for r in sub}) < 2 or len({r["shrink"] for r in rows}) < 10:
                continue
            m1 = [r["shrink"] for r in sub if r["arm"] == arms[0]]
            m2 = [r["shrink"] for r in sub if r["arm"] == arms[1]]
            pfs = [r["pfs"] for r in rows]
            mean_pfs = sum(pfs) / len(pfs)
            # the count above the MEAN varies per table (above the median of 10 values it
            # was always 5 in v0.2); reject tables with a value on the mean so "strictly
            # greater" is unambiguous across floating-point paths
            if any(abs(x - mean_pfs) < 1e-6 for x in pfs):
                continue
            means = {a: sum(r["shrink"] for r in rows if r["arm"] == a) / len([r for r in rows if r["arm"] == a])
                     for a in arms}
            best = max(means, key=means.get)
            expected = [
                f"{sum(m1) / len(m1) - sum(m2) / len(m2):.4f}",
                str(sum(x > mean_pfs for x in pfs)),
                f"{sum(r['pfs'] for r in rows if r['arm'] == best) / len([r for r in rows if r['arm'] == best]):.4f}",
                max(sub, key=lambda r: r["shrink"])["patient_id"],
            ][kind]
            break
        csv = "\n".join(["patient_id,arm,kras_status,tumor_shrinkage_pct,pfs_months"]
                        + [f"{r['patient_id']},{r['arm']},{r['kras']},{r['shrink']},{r['pfs']}" for r in rows])
        subst = dict(a1=arms[0], a2=arms[1], ks=ks)
        t_en, t_it, code = TOOLH_TASKS[kind]
        target = json.dumps({"tool": "execute_python", "arguments": {"code": code.format(**subst)}})
        for lang, tpl, task, out in (("en", TOOL_EN, t_en, en), ("it", TOOL_IT, t_it, it)):
            out.append(Item(id=f"crc-toolh-{lang}-{i:03d}", kind=Kind.PROGRAM,
                            prompt=tpl.format(csv=csv, task=task.format(**subst)),
                            target=target, checker="./checker.py:score_tool_call_hard",
                            # 300 (v0.3) truncated models that re-embed the table as a literal:
                            # scored as a malformed call, not as code ability
                            category=lang, tags=[f"expect:{expected}"], max_tokens=1200, provenance=PROV()))
    return en, it


# ----------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="suites/crc_pilot_v0.4")
    args = ap.parse_args()
    out = Path(args.out)

    mcq_en, mcq_it = build_mcq()
    ext_en, ext_it = build_extract()
    cmp_en, cmp_it = build_compute()
    sum_en, sum_it = build_summary()
    tool_en, tool_it = build_tool()
    mcqh_en, mcqh_it = build_mcq_hard()
    exth_en, exth_it = build_extract_hard()
    sumh_en, sumh_it = build_summary_hard()
    toolh_en, toolh_it = build_tool_hard()

    def task(name, kind, desc, **scoring):
        return Task(name=name, file=f"{name}.jsonl", kind=kind, description=desc, scoring=scoring)

    tasks = [
        task("mcq_en", Kind.CHOICE_GENERATE, "tier 1 EN: CRC drug/biomarker knowledge (hand-written, expert review pending)"),
        task("mcq_it", Kind.CHOICE_GENERATE, "tier 1 IT: parallel translations of mcq_en (pair: tags)"),
        # extraction reads the LAST number: a preamble echoing the question ("In Study 2, ...")
        # or a stray leading token is not read as the answer; IT tasks also accept a decimal
        # comma so locale does not masquerade as an extraction or arithmetic error
        task("extract_en", Kind.NUMERIC, "tier 2 EN: extract one number from a synthetic trial abstract", number="last"),
        task("extract_it", Kind.NUMERIC, "tier 2 IT: parallel abstracts", number="last", decimal_comma=True),
        task("compute_en", Kind.NUMERIC, "tier 3 EN: derive risk difference / NNT / pooled ORR",
             answer_prefix="ANSWER:"),
        task("compute_it", Kind.NUMERIC, "tier 3 IT: parallel (either marker accepted)",
             answer_prefix=["RISPOSTA:", "ANSWER:"], decimal_comma=True),
        task("summary_en", Kind.PROGRAM, "tier 4 EN: abstract -> fixed-schema JSON, scored by ./checker.py:score_summary", temperature=0.0),
        task("summary_it", Kind.PROGRAM, "tier 4 IT: parallel (JSON keys stay English)", temperature=0.0),
        task("tool_en", Kind.PROGRAM, "tier 5 EN: emit an execute_python tool call, scored by ./checker.py:score_tool_call", temperature=0.0),
        task("tool_it", Kind.PROGRAM, "tier 5 IT: parallel", temperature=0.0),
        # hard band: keeps discriminating where full-precision 27B models ace the base band
        task("mcq_hard_en", Kind.CHOICE_GENERATE, "hard tier 1 EN: trial attribution, metabolites, fine pharmacology (expert review pending)"),
        task("mcq_hard_it", Kind.CHOICE_GENERATE, "hard tier 1 IT: parallel"),
        task("extract_hard_en", Kind.NUMERIC, "hard tier 2 EN: two studies + subgroup rows; disambiguate study, arm and subgroup",
             number="last"),
        task("extract_hard_it", Kind.NUMERIC, "hard tier 2 IT: parallel", number="last", decimal_comma=True),
        task("summary_hard_en", Kind.PROGRAM, "hard tier 4 EN: schema adds derived field orr_absolute_difference_pct", temperature=0.0),
        task("summary_hard_it", Kind.PROGRAM, "hard tier 4 IT: parallel", temperature=0.0),
        task("tool_hard_en", Kind.PROGRAM, "hard tier 5 EN: emitted code is EXECUTED against the table; printed result must match", temperature=0.0),
        task("tool_hard_it", Kind.PROGRAM, "hard tier 5 IT: parallel", temperature=0.0),
    ]
    items = {"mcq_en": mcq_en, "mcq_it": mcq_it, "extract_en": ext_en, "extract_it": ext_it,
             "compute_en": cmp_en, "compute_it": cmp_it, "summary_en": sum_en, "summary_it": sum_it,
             "tool_en": tool_en, "tool_it": tool_it,
             "mcq_hard_en": mcqh_en, "mcq_hard_it": mcqh_it, "extract_hard_en": exth_en, "extract_hard_it": exth_it,
             "summary_hard_en": sumh_en, "summary_hard_it": sumh_it, "tool_hard_en": toolh_en, "tool_hard_it": toolh_it}
    meta = SuiteMeta(
        name="crc_pilot", version="0.4",
        description="SLM capability ladder for colorectal-cancer drug-efficacy work: knowledge -> extraction -> "
                    "computation -> structured summarization -> tool calls, each in EN and IT. Built to compare "
                    "model families and quantizations of the same family per tier.",
        tasks=tasks,
        meta={"domain": "colorectal cancer drug efficacy", "languages": ["en", "it"], "seed": SEED,
              "review_status": "DRAFT v0.4 — tier-1 facts (base + hard) and IT phrasing pending domain-expert review",
              "tiers": {"1": "mcq", "2": "extract", "3": "compute", "4": "summary", "5": "tool"},
              "hard_band": ["mcq_hard", "extract_hard", "summary_hard", "tool_hard"],
              "exec_warning": "tool_hard_* checkers execute model-generated Python in a subprocess (10s timeout)"},
    )
    save_suite(out, meta, items)
    for t in tasks:
        print(f"{t.name:12s} {len(items[t.name]):3d} items")
    dirs = Counter(tag for its in items.values() for it in its for tag in it.tags if tag.startswith("direction:"))
    print("efficacy direction of the synthetic abstracts:", dict(dirs))
    print("total", sum(len(v) for v in items.values()), "->", out)
    print("NOTE: checker.py must exist in the suite dir (it is part of the content hash).")


if __name__ == "__main__":
    main()
