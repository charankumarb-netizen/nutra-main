"""
Labelled ENGLISH clinical SENTENCES for training the context classifier.

Classes (one label per sentence):
    affirmed         the deficiency (or a low level) is stated as present          "Patient has iron deficiency."
    negated          the deficiency is absent / excluded, or the level is normal   "No iron deficiency." "Ferritin normal."
    suspected        possible / probable / to be investigated                      "Suspected vitamin B12 deficiency."
    family_history   it concerns a relative                                        "Mother with iron deficiency."
    historical       it was in the past / resolved                                 "History of iron deficiency, treated."
    hypothetical     prevention, risk, conditional, advice                         "Counseling to prevent iron deficiency."
    lab_measurement  a laboratory result line                                      "Ferritin: 8 ng/mL."
    unrelated        anything else, incl. HARD negatives that reuse the same words "No visual impairment." "Shortness of breath."

HOW THE DATA IS BUILT (honest description)
    Sentences come from hand-written templates x vocabulary x optional context x noise (accent loss, capitals,
    typos, OCR confusions). The label is decided by the TEMPLATE, before any text is generated, never by the model
    or by the rule engine. Templates are split into a TRAIN group and a held-out TEST group (every 4th cue of
    each class is held out), so the test score measures generalisation to unseen phrasing, not memorisation.
    This is synthetic data: it teaches the model the linguistic patterns; it does NOT replace real,
    clinician-annotated English notes (see docs/PERSON3_NLP_GUIDE.md, "Training on real data").
"""

from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass

LABELS = ["affirmed", "negated", "suspected", "family_history", "historical", "hypothetical", "lab_measurement", "unrelated"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

NUTRIENTS = [
    "iron", "vitamin D", "vitamin B12", "calcium", "folic acid", "folate", "cobalamin", "vitamin B 12",
    "vit D", "vit. B12", "25-OH vitamin D", "ferritin", "serum calcium", "vitamin D3", "B12", "serum iron",
]
DEF = ["deficiency", "insufficiency", "depletion", "deficit", "shortage", "lack"]
DEF_POST = ["deficiency", "insufficiency", "depletion"]
DEF_ADJ = ["severe", "mild", "moderate", "marked", "significant", "slight"]
RELATIVES = ["mother", "father", "sister", "brother", "grandmother", "grandfather", "aunt", "daughter", "son", "cousin"]
YEARS = ["2015", "2018", "2019", "2020", "2021", "2022"]
PREFIX = ["", "", "", "45-year-old patient. ", "Follow-up visit: ", "Note: ", "Today, ", "On examination, ", "As assessed, ", "Comment: "]
SUFFIX = ["", "", "", " Plan unchanged.", " Return in 30 days.", " New tests requested.", " No other complaints.", " Advice given."]

# content expressions for a LOW nutrient (used by every class except negated-normal and lab/unrelated)
CONTENT = [
    "{n} {dp}", "{n} {dp}", "{d} of {n}", "{a} {n} {dp}", "low levels of {n}", "reduced {n} levels",
    "decreased {n}", "low {n}", "{n} below the reference range", "{n} level reduced", "{n} insufficient",
    "{n} is low", "{n} was low", "low serum {n}",
    # disease names the rule engine also understands (no {n} placeholder)
    "iron deficiency anemia", "pernicious anemia", "megaloblastic anemia", "hypocalcemia", "hypovitaminosis D",
    "hypovitaminosis B12", "anemia due to iron deficiency", "{n} low ({v} {u})", "{n} reduced, {v} {u}",
]
# content expressions that state a NORMAL level (only used by the negated class)
CONTENT_NORMAL = [
    "{n} normal", "{n} within normal limits", "{n} levels adequate", "{n} within the reference range",
    "{n} preserved", "{n} unremarkable", "{n} level satisfactory", "{n} within the expected range",
]
# non-target conditions: hard negatives that reuse the same cue words
OTHER_CONDITIONS = [
    "hypertension", "diabetes", "asthma", "pneumonia", "chest pain", "fever", "visual impairment", "hearing deficit",
    "cognitive deficit", "lactase deficiency", "G6PD deficiency", "heart failure", "renal insufficiency",
    "attention deficit", "shortness of breath", "loss of appetite", "emotional deprivation", "gastritis",
    "urinary tract infection", "food allergy", "arthritis", "migraine", "obesity", "depression", "dyslipidemia",
    "fracture", "cough", "low back pain", "focal neurological deficit",
]

# class -> (pre cues, post cues)
_GRAMMAR: dict[str, tuple[list[str], list[str]]] = {
    "affirmed": (
        ["Patient has", "Diagnosis:", "Confirmed", "Documented", "Presents with", "Findings show", "Known", "Patient with",
         "Diagnosed with", "Persistent", "Ongoing", "Noted:", "Lab findings:", "Current:", "Has", "Shows", "Established",
         "Reports and confirmed", "Detected", "Clinically evident", "Active", "Patient presents", "Impression:", "Result:"],
        ["", "", "", "confirmed", "for several months", "persistent", "despite supplementation", "associated with anemia",
         "on replacement", "currently present", "evident", "ongoing", "symptomatic", "on today's labs"],
    ),
    "negated": (
        ["No", "Denies", "Without", "Absence of", "Ruled out", "Excluded", "Negative for", "No evidence of", "No signs of",
         "Patient denies", "Free of", "There is no", "No sign of", "Does not have", "Never had", "Not", "Patient has no",
         "No history of", "No clinical", "No known", "Neither", "No suggestion of", "Denied", "Without any"],
        ["", "", "", "ruled out", "excluded", "absent", "was not found", "not detected", "not confirmed by labs",
         "not present", "was not identified", "has been excluded", "not evident"],
    ),
    "suspected": (
        ["Suspected", "Possible", "Probable", "Cannot rule out", "Investigate", "Rule out", "Hypothesis:", "Suggestive of",
         "Query", "Maybe", "Consider", "Evaluate for", "Likely", "Possibly", "Cannot be excluded:", "Under investigation:",
         "Differential includes", "Work-up for", "Suspicion of", "May have", "Might be", "Presumed", "Appears to have",
         "Assess for", "Concern for", "Check for"],
        ["", "", "", "to be confirmed", "pending", "under investigation", "possible", "likely", "to rule out",
         "needs confirmation", "to be clarified", "unconfirmed"],
    ),
    "family_history": (
        ["{r} with", "{r} has", "{r} had", "{r} diagnosed with", "Family history of", "Family history:", "In the family:",
         "Positive family history of", "Relative with", "{r} and {r2} with", "{r} of the patient with", "{r} treated for",
         "{r} takes replacement for", "First-degree relative with", "Pedigree: {r} with", "Cases of", "{r} reports",
         "Father and {r} with", "Relatives with", "Maternal grandmother with", "{r} suffers from", "{r} presented with"],
        ["", "", "", "in the family", "in a first-degree relative", "on the maternal side", "on the paternal side",
         "in relatives", "in the {r}", "in the mother"],
    ),
    "historical": (
        ["History of", "Past history of", "Previously had", "Prior", "Formerly had", "Used to have", "Hx of", "h/o",
         "Previous episode of", "Was previously diagnosed with", "Treated for", "Remote", "Earlier", "In the past:",
         "Past medical history:", "Had previously", "Previous", "In {y} had", "Childhood", "Status post treatment of",
         "Was treated for", "Previously noted:", "Prior history of", "Resolved"],
        ["", "treated", "already resolved", "corrected with replacement", "in {y}", "normalized after treatment",
         "years ago", "in the past", "as a child", "has resolved", "in childhood", "after treatment", "now resolved"],
    ),
    "hypothetical": (
        ["Prevention of", "Risk of", "Avoid", "May cause", "If there is", "In case of", "To prevent", "Counseling to avoid",
         "Long-term use may lead to", "Predisposes to", "Increases the risk of", "Monitor for", "Should be monitored for",
         "When there is", "Risk factors for", "Prone to", "Prophylaxis against", "Unless there is", "At risk of",
         "Advice to avoid", "Strategies to prevent", "Whether there is", "Education on", "Might develop"],
        ["", "", "in vegetarians", "in older adults", "with a balanced diet", "during pregnancy", "if risk factors are present",
         "as advised", "in users of omeprazole", "after bariatric surgery", "if needed", "in restrictive diets"],
    ),
}


def _split_of(label: str, idx: int) -> str:
    """Every 4th cue of a class is held out for the TEST split."""
    return "test" if idx % 4 == 3 else "train"


def _cues(label: str, which: int, split: str) -> list[tuple[int, str]]:
    cues = _GRAMMAR[label][which]
    return [(i, c) for i, c in enumerate(cues) if c == "" or split == "all" or _split_of(label, i) == split]


_LAB_NAMES = [
    ("Ferritin", "ng/mL", (3, 400)), ("Vitamin D", "ng/mL", (8, 80)), ("Vitamin B12", "pg/mL", (120, 900)),
    ("Calcium", "mg/dL", (6.5, 11)), ("Hemoglobin", "g/dL", (7, 16)), ("Folate", "ng/mL", (1.5, 18)),
    ("Serum iron", "µg/dL", (20, 160)), ("25-OH Vitamin D", "ng/mL", (6, 70)), ("Cobalamin", "pg/mL", (100, 800)),
    ("Folic acid", "ng/mL", (1.5, 20)), ("Transferrin saturation", "%", (5, 45)), ("Ionized calcium", "mmol/L", (0.9, 1.4)),
]
_LAB_FORMATS = [
    "{l}: {v} {u}.", "{l} {v} {u}", "{l} ....... {v} {u}", "{l} = {v} {u}", "Result for {l}: {v} {u}.",
    "{l} - {v} {u}", "{l}: {v} {u} (Ref: {lo} - {hi})", "{l} {v}{u}", "{l} level: {v} {u}.", "{l}: {v} {u} [reference: {lo} - {hi}]",
]
_UNRELATED = [
    "Patient reports a headache for two days.", "Blood pressure 120/80 mmHg.", "Return in 30 days.",
    "Physical examination unremarkable.", "Denies drug allergies.", "No fever.", "No visual impairment.",
    "Mild hearing deficit on the right.", "Mother with diabetes.", "History of hypertension.",
    "Suspected pneumonia.", "High fall risk.", "Low-sodium diet advised.", "Shortness of breath on exertion.",
    "Attention deficit under follow-up.", "Emotional deprivation reported by the family.", "Lactase deficiency confirmed.",
    "No focal neurological deficit.", "Possible urinary tract infection.", "Father with myocardial infarction at 55.",
    "Prescribed paracetamol 750 mg as needed.", "Female patient, 45 years old.", "Collection date: 03/12/2024.",
    "License no. 123456.", "Lungs clear on auscultation.", "Abdomen soft, non-tender.", "Denies smoking and alcohol use.",
    "Loss of appetite in recent days.", "Had chickenpox in childhood.", "Avoid strenuous exercise.",
    "No leg edema.", "Heart rate 78 bpm.", "Fasting glucose 92 mg/dL.",
    "Total cholesterol 190 mg/dL.", "Allergy to penicillin.", "Appendectomy in 2015.", "Abdominal ultrasound requested.",
    "Physical activity counseling given.", "Heart failure compensated.", "G6PD deficiency ruled out.",
    "Mother with hypertension.", "Intermediate cardiovascular risk.", "No urinary complaints.", "Possible food allergy.",
    "Working diagnosis: gastritis.", "Previously had asthma, controlled.", "Shortness of breath when lying down.",
    "Patient counseled on medication use.", "Reports tiredness after work.", "White blood cells 7,200 /mm3.",
    "Glycated hemoglobin 6.5 %.", "No history of trauma.", "Dry cough for one week.", "Usual diet of rice and beans.",
    "Weight 68 kg, height 1.65 m.", "Attending physician: Dr. Smith.", "Plan: continue outpatient follow-up.",
    "Chronic kidney disease stage 2.", "Mild cognitive deficit under investigation.", "Brother with asthma.",
    "Had dengue in 2019.", "If fever, seek care.", "Fall prevention advised.",
]


@dataclass(frozen=True)
class Example:
    text: str
    label: str
    template: str       # "<label>#<index>" - lets you audit leakage between splits
    split: str


# ---------------------------------------------------------------------------
# Noise (what real reports / OCR do to text)
# ---------------------------------------------------------------------------


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def add_noise(text: str, rng: random.Random) -> str:
    r = rng.random()
    if r < 0.18:
        text = _strip_accents(text)
    elif r < 0.30:
        text = text.upper()
    elif r < 0.40:
        text = text.lower()
    elif r < 0.48 and len(text) > 12:                    # one-character typo
        i = rng.randrange(1, len(text) - 2)
        text = text[:i] + text[i + 1] + text[i] + text[i + 2:]
    elif r < 0.54:
        text = text.replace("B12", "B 12").replace("ng/mL", "ng/m1")
    elif r < 0.58:
        text = re.sub(r"[.,;:]", "", text)
    return text


def _lab_line(rng: random.Random) -> str:
    name, unit, (lo, hi) = rng.choice(_LAB_NAMES)
    v = rng.uniform(lo, hi)
    val = f"{v:.0f}" if v > 30 else f"{v:.1f}"
    fmt = rng.choice(_LAB_FORMATS)
    return fmt.format(l=name, v=val, u=unit, lo=rng.choice(["30", "20", "200", "8.5", "12"]), hi=rng.choice(["100", "400", "900", "10.5", "16"]))


def _fmt(text: str, rng: random.Random, n: str | None = None) -> str:
    d = rng.choice(DEF)
    return text.format(d=d, dp=rng.choice(DEF_POST), n=n or rng.choice(NUTRIENTS), r=rng.choice(RELATIVES), r2=rng.choice(RELATIVES),
                       y=rng.choice(YEARS), a=rng.choice(DEF_ADJ), v=rng.choice(["8", "12", "18", "130", "6.5", "1.9"]),
                       u=rng.choice(["ng/mL", "pg/mL", "mg/dL", "g/dL"]))


def _compose(label: str, rng: random.Random, split: str, content: str) -> str:
    """pre cue + content + post cue of a class, e.g.  'Suspected' + 'iron deficiency' + 'to be confirmed'."""
    pre = rng.choice(_cues(label, 0, split))[1]
    post = rng.choice(_cues(label, 1, split))[1]
    if label == "affirmed" and rng.random() < 0.12:
        pre = ""
    return " ".join(p for p in (pre, content, post) if p)


def _sentence(label: str, rng: random.Random, split: str) -> tuple[str, str]:
    """-> (text, template id). The id names the kind of sentence, to audit the train/test split."""
    if label == "negated" and rng.random() < 0.30:                    # "ferritin normal" style: a normal level
        content = rng.choice(CONTENT_NORMAL)
        pre = rng.choice(["", "", "", "Tests show", "It is noted that", "Result:", "Patient with", "Today:"])
        post = rng.choice([c for _, c in _cues("affirmed", 1, split)])
        text = _fmt(" ".join(x for x in (pre, content, post) if x), rng)
        return text, "negated#normal-level"
    if label == "unrelated":
        if rng.random() < 0.55:                                         # hard negative: same cue words, non-target condition
            cue_label = rng.choice(list(_GRAMMAR))
            cond = rng.choice(OTHER_CONDITIONS)
            text = _fmt(_compose(cue_label, rng, split, cond), rng)
            return text, f"unrelated#hardneg-{cue_label}"
        pool = [(i, t) for i, t in enumerate(_UNRELATED) if split == "all" or _split_of("unrelated", i) == split]
        i, text = rng.choice(pool)
        return text, f"unrelated#{i}"
    if label == "lab_measurement":
        return _lab_line(rng), "lab#generated"
    text = _fmt(_compose(label, rng, split, rng.choice(CONTENT)), rng)
    return text, f"{label}#grammar"


def generate_dataset(n_per_class: int = 900, seed: int = 0, split: str = "train", noise: float = 0.3,
                     context: float = 0.30) -> list[Example]:
    """
    Generate a balanced dataset.

    split='train' uses 3/4 of the cue phrases of every class, split='test' the held-out 1/4 (never seen in training).
    'lab_measurement' has no cue phrases (its numbers/analytes are random in both splits) and 'unrelated' uses the same
    train/test cue split for its hard negatives plus a split pool of generic sentences.
    """
    if split not in ("train", "test", "all"):
        raise ValueError("split must be 'train', 'test' or 'all'")
    rng = random.Random(f"{seed}:{split}")
    data: list[Example] = []
    for label in LABELS:
        for _ in range(n_per_class):
            text, tid = _sentence(label, rng, split)
            text = text[0].upper() + text[1:] if rng.random() < 0.9 else text
            if not text.endswith((".", ")", "]")) and rng.random() < 0.75 and label != "lab_measurement":
                text += "."
            if label not in ("lab_measurement",) and rng.random() < context:
                text = rng.choice(PREFIX) + text + rng.choice(SUFFIX)
            if rng.random() < noise:
                text = add_noise(text, rng)
            data.append(Example(text.strip(), label, tid, split))
    rng.shuffle(data)
    return data
