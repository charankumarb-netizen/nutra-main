"""
Synthetic English test-case generator with GOLD labels.

The gold label is decided FIRST (e.g. "vitamin D is deficient"), and only then a
value/sentence consistent with it is generated.  The engine is therefore tested
against ground truth that does not come from the engine itself.

Used by: tests (thousands of cases), scripts/evaluate.py, and as bootstrap data
for fine-tuning a transformer (scripts/finetune_classifier.py).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

ALL_KEYS = ["iron", "vitamin_d", "vitamin_b12", "calcium", "folate", "anemia"]

# analyte -> (names, unit variants [(unit_text, factor_from_canonical)], ranges per category)
_LAB = {
    "iron": dict(
        names=["Ferritin", "FERRITIN", "ferritin", "Serum ferritin"],
        units=[("ng/mL", 1.0), ("ng/ml", 1.0), ("ug/L", 1.0), ("µg/L", 1.0)],
        ranges={"deficient": (1.0, 27.0), "normal": (35.0, 300.0)},
    ),
    "vitamin_d": dict(
        names=["Vitamin D", "25-OH Vitamin D", "Vitamin D (25-OH)", "25-hydroxyvitamin D",
               "Vit D", "VITAMIN D3", "Vitamin D 25-OH"],
        units=[("ng/mL", 1.0), ("ng/ml", 1.0), ("nmol/L", 2.496)],
        ranges={"deficient": (3.0, 17.5), "borderline": (21.5, 28.0), "normal": (33.0, 90.0)},
    ),
    "vitamin_b12": dict(
        names=["Vitamin B12", "Vitamin B-12", "Cobalamin", "B12", "VITAMIN B12", "Vit B12"],
        units=[("pg/mL", 1.0), ("pg/ml", 1.0), ("pmol/L", 1 / 1.355)],
        ranges={"deficient": (60.0, 190.0), "borderline": (210.0, 290.0), "normal": (320.0, 900.0)},
    ),
    "folate": dict(
        names=["Folic acid", "Folate", "FOLIC ACID"],
        units=[("ng/mL", 1.0), ("ng/ml", 1.0), ("nmol/L", 2.266)],
        ranges={"deficient": (0.8, 2.6), "borderline": (3.5, 5.5), "normal": (7.0, 20.0)},
    ),
    "calcium": dict(
        names=["Calcium", "Total calcium", "Serum calcium", "CALCIUM"],
        units=[("mg/dL", 1.0), ("mg/dl", 1.0), ("mmol/L", 1 / 4.008)],
        ranges={"deficient": (6.5, 8.2), "normal": (8.8, 10.2)},
    ),
    "anemia": dict(
        names=["Hemoglobin", "Hb", "HEMOGLOBIN", "Haemoglobin"],
        units=[("g/dL", 1.0), ("g/dl", 1.0)],
        ranges={"deficient": (7.0, 11.5), "normal": (13.5, 16.5)},
    ),
}

_NAME_TEXT = {  # (noun, kept for compatibility with the old tuple shape)
    "iron": ("iron", "m"), "vitamin_d": ("vitamin D", "f"), "vitamin_b12": ("vitamin B12", "f"),
    "calcium": ("calcium", "m"), "folate": ("folic acid", "m"),
}

_TEXT_BY_STANCE = {
    "affirmed": [
        "Patient has {n} deficiency.", "Diagnosis: {n} deficiency.", "Deficiency of {n}.",
        "Picture of {n} deficiency.", "Presents with low levels of {n}.", "{N} low.", "Marked {n} deficiency.",
        "Patient with {n} deficiency, under treatment.",
    ],
    "denied": [
        "No {n} deficiency.", "Denies {n} deficiency.", "Does not have {n} deficiency.",
        "Ruled out {n} deficiency.", "Absence of {n} deficiency.", "There is no {n} deficiency.",
        "{N} normal.", "{N} within normal limits.", "{N} deficiency ruled out.",
    ],
    "uncertain": [
        "Suspected {n} deficiency.", "Possible {n} deficiency.", "Investigate {n} deficiency.",
        "Cannot rule out {n} deficiency.", "Working hypothesis: {n} deficiency.",
        "Test requested to rule out {n} deficiency.",
    ],
    "none_family": ["Mother with {n} deficiency.", "Family history of {n} deficiency."],
    "none_hypo": ["Risk of {n} deficiency.", "Diet advice to prevent {n} deficiency."],
    "none_hist": ["History of {n} deficiency in childhood.", "{N} deficiency treated."],
}

_DISTRACTORS = [
    "Glycated hemoglobin: 6.5 %", "Urine calcium: 250 mg/24h", "Collection date: 12/03/2024",
    "Patient: Maria Silva, 45 years", "Vitamin D3 50,000 IU weekly", "Ferrous sulfate 40 mg/day",
    "Total iron-binding capacity: 350 µg/dL", "Glucose: 92 mg/dL", "Total cholesterol: 190 mg/dL",
    "Leukocytes: 7,200 /mm3", "Creatinine: 0.9 mg/dL", "License 123456", "File no. 10.2",
]


@dataclass
class Case:
    text: str
    expected: dict[str, str]   # key -> status (all 6 keys)
    sex: str | None = None


def _fmt_value(v: float, rng: random.Random) -> str:
    style = rng.choice(["dot1", "dot2", "dot1", "int"])
    if style == "int" and v >= 20:
        return str(int(round(v)))
    if style == "dot2":
        return f"{v:.2f}"
    return f"{v:.1f}"


def _lab_line(key: str, cat: str, rng: random.Random, sex: str | None) -> str:
    spec = _LAB[key]
    lo, hi = spec["ranges"][cat]
    if key == "anemia" and cat == "normal" and sex == "M":
        lo, hi = 13.6, 17.0
    canonical = rng.uniform(lo, hi)
    unit, factor = rng.choice(spec["units"])
    # keep values safely away from thresholds even after rounding
    value = canonical * factor
    val_s = _fmt_value(value, rng)
    # if rounding to int moved a value close to a threshold we generated far from it already
    name = rng.choice(spec["names"])
    sep = rng.choice([": ", " ", " ....... ", " = ", " of ", " - "])
    if sep == " of ":
        name = name.lower() if rng.random() < 0.5 else name
    unit_sep = rng.choice([" ", "", " "])
    line = f"{name}{sep}{val_s}{unit_sep}{unit}"
    if rng.random() < 0.25:
        line += rng.choice([" (Reference range: 30 - 100 ng/mL)", " (Ref: 15 - 150)", " [ref.: > 20]"])
    if rng.random() < 0.1:
        line = line.upper()
    return line


def generate_case(rng: random.Random) -> Case:
    sex = rng.choice(["M", "F", None])
    expected = {k: "not_assessed" for k in ALL_KEYS}
    lines: list[str] = []
    if rng.random() < 0.7:
        lines.append(rng.choice(["MEDICAL REPORT", "Laboratory tests", "Report", "Results"]))
    if sex and rng.random() < 0.8:
        lines.append("Sex: " + ("Male" if sex == "M" else "Female"))

    keys = rng.sample(ALL_KEYS, rng.randint(1, len(ALL_KEYS)))
    for key in keys:
        cats = list(_LAB[key]["ranges"])
        mode = rng.choice(["lab", "lab", "text"]) if key != "anemia" else "lab"
        if mode == "lab":
            cat = rng.choice(cats)
            lines.append(_lab_line(key, cat, rng, sex if key == "anemia" else None))
            expected[key] = {"deficient": "deficient", "borderline": "borderline", "normal": "not_deficient"}[cat]
            if key == "anemia" and sex is None and cat == "normal":
                # unknown sex uses 12 g/dL; a normal value is > 13.5 so still normal
                pass
            if key != "anemia" and rng.random() < 0.35:
                n, g = _NAME_TEXT[key]
                if cat == "deficient":
                    stance = "affirmed"
                elif cat == "normal":
                    stance = "denied"
                else:
                    stance = None
                if stance:
                    tpl = rng.choice(_TEXT_BY_STANCE[stance])
                    lines.append(_render(tpl, n, g))
        else:
            n, g = _NAME_TEXT[key]
            kind = rng.choice(["affirmed", "denied", "uncertain", "none_family", "none_hypo", "none_hist"])
            tpl = rng.choice(_TEXT_BY_STANCE[kind])
            lines.append(_render(tpl, n, g))
            expected[key] = {"affirmed": "deficient", "denied": "not_deficient", "uncertain": "suspected"}.get(
                kind, "not_assessed")
    for _ in range(rng.randint(0, 3)):
        d = rng.choice(_DISTRACTORS)
        # distractors must not collide with a generated analyte line of the same kind
        lines.append(d)
    rng.shuffle(lines)
    text = "\n".join(lines)
    return Case(text=text, expected=expected, sex=sex)


def _render(tpl: str, n: str, g: str) -> str:
    N = n[0].upper() + n[1:]
    return tpl.format(n=n, N=N)


def generate_cases(count: int, seed: int = 42) -> list[Case]:
    rng = random.Random(seed)
    return [generate_case(rng) for _ in range(count)]
