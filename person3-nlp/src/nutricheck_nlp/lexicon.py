"""
NutriCheck - NLP engine.
Clinical lexicon: nutrients, laboratory analytes, units and reference ranges (ENGLISH).

All text patterns are written for NORMALISED text (lower-case, accents stripped,
see text_normalizer.py).  Thresholds are adult, general-population values and
are intentionally kept in ONE place so that a clinician can review/adjust them.

Language scope: English clinical text only.  Nothing in this module claims support
for any other language; text in other languages is detected (see LANGUAGE_MARKERS)
only so that the system can REFUSE to interpret it instead of guessing.

IMPORTANT: this is decision-SUPPORT software. It does not diagnose.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .multilingual import MULTILINGUAL_BIOMARKERS

# ---------------------------------------------------------------------------
# Targets (what the system reports on)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Target:
    key: str
    name: str
    kind: str  # "nutrient" | "finding"
    # regex (normalised text) for naming the target inside prose
    aliases: str
    recommendation: str


TARGETS: dict[str, Target] = {
    "iron": Target(
        "iron", "Iron", "nutrient",
        MULTILINGUAL_BIOMARKERS["ferritin"].compile_regex_pattern() + "|" +
        MULTILINGUAL_BIOMARKERS["serum_iron"].compile_regex_pattern() + "|" +
        MULTILINGUAL_BIOMARKERS["tsat"].compile_regex_pattern() +
        r"|(?:serum\s+|plasma\s+)?iron(?:\s+(?:serum|plasma))?(?![\s-]+(?:binding|saturation))|ferritin|iron\s+(?:stores|reserves)|ferro\s+corporal|reserva\s+de\s+ferro",
        "Confirm with ferritin and transferrin saturation; consider the cause (blood loss, diet, absorption).",
    ),
    "vitamin_d": Target(
        "vitamin_d", "Vitamin D", "nutrient",
        MULTILINGUAL_BIOMARKERS["vitamin_d"].compile_regex_pattern() +
        r"|(?:25\s*[-(]?\s*(?:oh|hydroxy)\s*\)?\s*-?\s*)?vit(?:amin|\.)?\s*d\s*3?(?:\s*,?\s*[-(]?\s*25\s*[-(]?\s*(?:oh|hydroxy\w*)\s*\)?)?|calcidiol|25\s*\(\s*oh\s*\)\s*d",
        "Replacement is a clinical decision made according to protocol; re-check 25-OH vitamin D after treatment.",
    ),
    "vitamin_b12": Target(
        "vitamin_b12", "Vitamin B12", "nutrient",
        MULTILINGUAL_BIOMARKERS["vitamin_b12"].compile_regex_pattern() +
        r"|vit(?:amin|\.)?\s*b[\s\-]?12|cyanocobalamin|cobalamin|\bb[\s\-]?12",
        "Look for the cause (diet, metformin, PPIs, atrophic gastritis); consider methylmalonic acid / homocysteine if borderline.",
    ),
    "calcium": Target(
        "calcium", "Calcium", "nutrient",
        MULTILINGUAL_BIOMARKERS["calcium_total"].compile_regex_pattern() + "|" +
        MULTILINGUAL_BIOMARKERS["calcium_ionized"].compile_regex_pattern() + "|" +
        MULTILINGUAL_BIOMARKERS["calcium_corrected"].compile_regex_pattern() +
        r"|(?:(?:total|serum|ionized|ionised|corrected|free)\s+)?calcium(?:\s*,?\s*(?:total|serum|ionized|ionised|corrected))?|calcemia",
        "Check albumin, PTH, vitamin D and kidney function before any replacement.",
    ),
    "folate": Target(
        "folate", "Folate (folic acid)", "nutrient",
        MULTILINGUAL_BIOMARKERS["folate"].compile_regex_pattern() +
        r"|folic\s+acid|folates?(?!\s+(?:rbc|red\s+cell))|vit(?:amin|\.)?\s*b[\s\-]?9",
        "Review diet, alcohol and medicines; consider measuring B12 together before replacing folate.",
    ),
    # "finding" targets are supporting findings, not nutrients
    "anemia": Target(
        "anemia", "Anemia", "finding",
        MULTILINGUAL_BIOMARKERS["hemoglobin"].compile_regex_pattern() +
        r"|h(?:a)?emoglobin(?!\s*(?:a1c|a\s*1\s*c|f\b|fetal))|\bhb\b(?!\s*a1c)|\bhgb\b",
        "Anemia has many causes; correlate with ferritin, B12, folate and the clinical picture.",
    ),
}

NUTRIENT_KEYS = [k for k, t in TARGETS.items() if t.kind == "nutrient"]
FINDING_KEYS = [k for k, t in TARGETS.items() if t.kind == "finding"]

# ---------------------------------------------------------------------------
# Laboratory analytes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Analyte:
    key: str
    target: str                      # TARGETS key this analyte informs
    label: str
    name_regex: str                  # regex for naming the analyte in a lab line
    canonical_unit: str
    units: dict[str, float]          # normalised unit -> factor to canonical unit
    deficient_below: float           # value < this  -> "deficient" (low)
    borderline_below: float | None   # value < this  -> "borderline"
    high_above: float | None         # value > this  -> "high"
    plausible: tuple[float, float]   # physiologic plausibility (OCR error guard)
    tier: int = 1                    # lower tier = more authoritative
    weak: bool = False               # weak evidence (e.g. serum iron alone)
    default_unit: str | None = None  # used (with a warning, never for a final decision) when unit is missing
    qualifier_exclusions: str = ""   # regex lookahead exclusions (text AFTER the name)
    before_exclusions: str = ""      # regex tested against the ~24 characters BEFORE the name


_CA_BEFORE_EXCL = r"(?:(?:ionized|ionised|free|corrected|urine|urinary|albumin[\s-]*corrected)|(?<![/\w])24[\s-]*h(?:our)?s?[ \t]+(?:urine[ \t]+)?)[ \t,:-]*$"

ANALYTES: dict[str, Analyte] = {
    "ferritin": Analyte(
        "ferritin", "iron", "Ferritin",
        MULTILINGUAL_BIOMARKERS["ferritin"].compile_regex_pattern() + r"|ferritin",
        "ng/ml", {"ng/ml": 1.0, "ug/l": 1.0},
        30.0, None, 500.0, (0.1, 50000.0), tier=1, default_unit="ng/ml",
    ),
    "tsat": Analyte(
        "tsat", "iron", "Transferrin saturation",
        MULTILINGUAL_BIOMARKERS["tsat"].compile_regex_pattern() +
        r"|transferrin\s+saturation(?:\s+index)?|saturation\s+of\s+transferrin|iron\s+saturation(?:\s+index)?|\btsat\b|\btsa?t\b|\bist\b",
        "%", {"%": 1.0},
        20.0, None, 50.0, (0.5, 100.0), tier=1, default_unit="%",
    ),
    "serum_iron": Analyte(
        "serum_iron", "iron", "Serum iron",
        MULTILINGUAL_BIOMARKERS["serum_iron"].compile_regex_pattern() +
        r"|(?:serum\s+|plasma\s+)?iron(?:\s*,?\s*(?:serum|plasma))?",
        "ug/dl", {"ug/dl": 1.0, "umol/l": 5.585},
        60.0, None, 170.0, (1.0, 1000.0), tier=2, weak=True, default_unit="ug/dl",
        qualifier_exclusions=r"(?![\s-]*(?:binding|saturation|studies|deficien\w*|supplement\w*|tablets?|sulfate|stores|reserves|overload))",
        before_exclusions=r"(?:total|unsaturated|latent)[ \t]*$",
    ),
    "vitamin_d": Analyte(
        "vitamin_d", "vitamin_d", "25-OH Vitamin D",
        MULTILINGUAL_BIOMARKERS["vitamin_d"].compile_regex_pattern() + r"|" + TARGETS["vitamin_d"].aliases,
        "ng/ml", {"ng/ml": 1.0, "ug/l": 1.0, "nmol/l": 1 / 2.496},
        20.0, 30.0, 100.0, (0.5, 500.0), tier=1, default_unit="ng/ml",
    ),
    "vitamin_b12": Analyte(
        "vitamin_b12", "vitamin_b12", "Vitamin B12",
        MULTILINGUAL_BIOMARKERS["vitamin_b12"].compile_regex_pattern() + r"|" + TARGETS["vitamin_b12"].aliases,
        "pg/ml", {"pg/ml": 1.0, "ng/l": 1.0, "pmol/l": 1.355, "ng/ml": 1000.0},
        200.0, 300.0, 2000.0, (10.0, 200000.0), tier=1, default_unit="pg/ml",
    ),
    "folate": Analyte(
        "folate", "folate", "Folate",
        MULTILINGUAL_BIOMARKERS["folate"].compile_regex_pattern() + r"|" + TARGETS["folate"].aliases,
        "ng/ml", {"ng/ml": 1.0, "ug/l": 1.0, "nmol/l": 0.4413},
        3.0, 6.0, 24.0, (0.1, 1000.0), tier=1, default_unit="ng/ml",
        before_exclusions=r"(?:rbc|red\s*cell|red\s*blood\s*cell|erythrocyte)[ \t]*$",
    ),
    "calcium_ionized": Analyte(
        "calcium_ionized", "calcium", "Ionized calcium",
        MULTILINGUAL_BIOMARKERS["calcium_ionized"].compile_regex_pattern() +
        r"|(?:ionized|ionised|free)\s+calcium|calcium\s*,?\s*(?:ionized|ionised|free)",
        "mmol/l", {"mmol/l": 1.0, "mg/dl": 0.2495, "meq/l": 0.5},
        1.12, None, 1.32, (0.2, 3.0), tier=1, default_unit="mmol/l",
    ),
    "calcium_corrected": Analyte(
        "calcium_corrected", "calcium", "Corrected calcium",
        MULTILINGUAL_BIOMARKERS["calcium_corrected"].compile_regex_pattern() +
        r"|(?:albumin[\s-]*)?corrected\s+calcium|calcium\s*,?\s*corrected(?:\s+(?:for|by)\s+albumin)?",
        "mg/dl", {"mg/dl": 1.0, "mmol/l": 4.008, "meq/l": 2.004, "mg/l": 0.1},
        8.5, None, 10.5, (2.0, 30.0), tier=2, default_unit="mg/dl",
    ),
    "calcium_total": Analyte(
        "calcium_total", "calcium", "Total calcium",
        MULTILINGUAL_BIOMARKERS["calcium_total"].compile_regex_pattern() +
        r"|(?:(?:total|serum)\s+)?calcium(?:\s*,?\s*(?:total|serum))?|calcemia",
        "mg/dl", {"mg/dl": 1.0, "mmol/l": 4.008, "meq/l": 2.004, "mg/l": 0.1},
        8.5, None, 10.5, (2.0, 30.0), tier=3, default_unit="mg/dl",
        qualifier_exclusions=r"(?!\s*,?\s*(?:ionized|ionised|free|urine|urinary|corrected|24|ioni))",
        before_exclusions=_CA_BEFORE_EXCL,
    ),
    "hemoglobin": Analyte(
        "hemoglobin", "anemia", "Hemoglobin",
        MULTILINGUAL_BIOMARKERS["hemoglobin"].compile_regex_pattern() +
        r"|h(?:a)?emoglobin(?!\s*(?:a1c|a\s*1\s*c|f\b|fetal))|\bhb\b(?!\s*a1c)|\bhgb\b",
        "g/dl", {"g/dl": 1.0, "g/l": 0.1},
        12.0, None, None, (1.0, 25.0), tier=1, default_unit="g/dl",
        before_exclusions=r"(?:glycated|glycosylated|glyco|fetal|mean\s+corpuscular|mean\s+cell|corpuscular)[ \t]*$",
    ),
    # Additional CBC / Hematology measurements
    "mcv": Analyte(
        "mcv", "anemia", "MCV (Mean Corpuscular Volume)",
        MULTILINGUAL_BIOMARKERS["mcv"].compile_regex_pattern(),
        "fl", {"fl": 1.0},
        80.0, None, 100.0, (40.0, 160.0), tier=2, default_unit="fl",
    ),
    "hematocrit": Analyte(
        "hematocrit", "anemia", "Hematocrit",
        MULTILINGUAL_BIOMARKERS["hematocrit"].compile_regex_pattern(),
        "%", {"%": 1.0},
        36.0, None, 50.0, (10.0, 75.0), tier=2, default_unit="%",
    ),
    "mch": Analyte(
        "mch", "anemia", "MCH (Mean Corpuscular Hemoglobin)",
        MULTILINGUAL_BIOMARKERS["mch"].compile_regex_pattern(),
        "pg", {"pg": 1.0},
        27.0, None, 33.0, (10.0, 60.0), tier=2, default_unit="pg",
    ),
    "mchc": Analyte(
        "mchc", "anemia", "MCHC (Mean Corpuscular Hemoglobin Concentration)",
        MULTILINGUAL_BIOMARKERS["mchc"].compile_regex_pattern(),
        "g/dl", {"g/dl": 1.0, "%": 1.0},
        32.0, None, 36.0, (20.0, 45.0), tier=2, default_unit="g/dl",
    ),
}

# Hemoglobin anemia thresholds (WHO, adults, non-pregnant)
HEMOGLOBIN_THRESHOLD_BY_SEX = {"M": 13.0, "F": 12.0, None: 12.0}

# Tiers per target: analyte keys grouped by authority.  The first tier that has
# at least one usable result decides.  ("weak" analytes can only SUSPECT.)
TARGET_TIERS: dict[str, list[list[str]]] = {
    "iron": [["ferritin", "tsat"], ["serum_iron"]],
    "vitamin_d": [["vitamin_d"]],
    "vitamin_b12": [["vitamin_b12"]],
    "folate": [["folate"]],
    "calcium": [["calcium_ionized"], ["calcium_corrected"], ["calcium_total"]],
    "anemia": [["hemoglobin"], ["mcv", "hematocrit", "mch", "mchc"]],
}

# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

# units that mean "this is a medication dose, not a lab result"
DOSE_UNIT_REGEX = (
    r"\s*(?:iu|i\.u\.|units?|mg|mcg|ug|g|drops?|tablets?|tabs?|capsules?|caps|ampoules?|vials?|ml)\b"
    r"(?!\s*/\s*(?:dl|l|ml)\b)"
)

# the unit that may follow a number in a lab line (always has "/" except %, fl and pg)
UNIT_REGEX = r"(?:%|[a-zµμ]{1,6}\s*/\s*[a-z0-9]{1,3}|fl|pg)"

# ---------------------------------------------------------------------------
# Symptoms / risk factors / treatments  (supporting hints only - never decide)
# ---------------------------------------------------------------------------

SYMPTOMS: list[tuple[str, str, list[str]]] = [
    ("fatigue / tiredness", r"fatigue|tiredness|tired|asthenia|exhaustion|lethargy", ["iron", "vitamin_b12", "vitamin_d", "folate"]),
    ("weakness", r"weakness|feeling\s+weak", ["iron", "vitamin_b12", "vitamin_d"]),
    ("pallor", r"pallor|pale|paleness", ["iron", "vitamin_b12", "folate"]),
    ("hair loss", r"hair\s+loss|alopecia|hair\s+thinning", ["iron"]),
    ("brittle nails", r"brittle\s+nails?|koilonychia|spoon\s+nails?", ["iron"]),
    ("pica / pagophagia", r"pagophagia|pica|craving\s+ice|ice\s+craving", ["iron"]),
    ("restless legs", r"restless\s+legs?(?:\s+syndrome)?", ["iron"]),
    ("breathlessness on exertion", r"dyspn[o]?ea|shortness\s+of\s+breath|breathlessness", ["iron"]),
    ("tingling / paresthesia", r"tingling|paresthesias?|paraesthesias?|numbness|pins\s+and\s+needles", ["vitamin_b12"]),
    ("neuropathy", r"neuropathy|peripheral\s+neuropathy", ["vitamin_b12"]),
    ("glossitis", r"glossitis|smooth\s+tongue|sore\s+tongue", ["vitamin_b12", "iron", "folate"]),
    ("memory problems", r"memory\s+(?:loss|problems?|impairment)|forgetfulness", ["vitamin_b12"]),
    ("bone pain", r"bone\s+pain|osteomalacia|aching\s+bones", ["vitamin_d", "calcium"]),
    ("muscle pain / weakness", r"myalgia|muscle\s+(?:pain|aches?|weakness)", ["vitamin_d"]),
    ("cramps / tetany", r"cramps?|tetany|muscle\s+spasms?", ["calcium"]),
    ("osteoporosis / osteopenia", r"osteoporosis|osteopenia|fragility\s+fractures?|pathological\s+fractures?", ["vitamin_d", "calcium"]),
    ("mouth ulcers", r"stomatitis|mouth\s+ulcers?|oral\s+ulcers?|aphthous\s+ulcers?", ["folate", "vitamin_b12", "iron"]),
]

RISK_FACTORS: list[tuple[str, str, list[str]]] = [
    ("vegetarian / vegan diet", r"vegetarian|vegan|vegetarianism|veganism|plant[\s-]based\s+diet", ["vitamin_b12", "iron"]),
    ("bariatric surgery / gastrectomy", r"bariatric\s+surgery|gastrectomy|gastric\s+bypass|weight[\s-]loss\s+surgery|sleeve\s+gastrectomy", ["vitamin_b12", "iron", "calcium", "folate", "vitamin_d"]),
    ("metformin use", r"metformin", ["vitamin_b12"]),
    ("proton pump inhibitor use", r"omeprazole|pantoprazole|esomeprazole|lansoprazole|proton[\s-]pump\s+inhibitors?|\bppis?\b", ["vitamin_b12", "iron", "calcium"]),
    ("atrophic gastritis / pernicious anemia", r"atrophic\s+gastritis|pernicious\s+an(?:a)?emia", ["vitamin_b12", "iron"]),
    ("celiac disease / malabsorption", r"celiac|coeliac|malabsorption|crohn(?:'?s)?(?:\s+disease)?", ["iron", "vitamin_b12", "folate", "calcium", "vitamin_d"]),
    ("heavy menstrual bleeding", r"menorrhagia|heavy\s+(?:menstrual\s+bleeding|periods?|menses)", ["iron"]),
    ("pregnancy", r"pregnan(?:t|cy)|gestation", ["iron", "folate", "calcium"]),
    ("alcohol use disorder", r"alcoholism|alcohol\s+use\s+disorder|heavy\s+alcohol(?:\s+use)?|alcohol\s+abuse", ["folate", "vitamin_b12"]),
    ("limited sun exposure", r"(?:limited|low|little|no|poor)\s+sun(?:light)?\s+exposure", ["vitamin_d"]),
    ("chronic kidney disease", r"chronic\s+kidney\s+disease|\bckd\b|renal\s+(?:insufficiency|failure)|kidney\s+failure", ["vitamin_d", "calcium"]),
    ("hypoparathyroidism", r"hypoparathyroidism", ["calcium"]),
    ("chronic corticosteroid use", r"corticosteroids?|glucocorticoids?|prednisone|prednisolone", ["vitamin_d", "calcium"]),
]

# drug / supplement names -> nutrient (informational "treatments" list)
TREATMENT_NAMES: list[tuple[str, list[str]]] = [
    (r"ferrous\s+(?:sulfate|sulphate|fumarate|gluconate)|(?:oral|intravenous|iv)\s+iron|iron\s+(?:supplements?|tablets?|infusions?|pills?|polymaltose)", ["iron"]),
    (r"cholecalciferol|calcitriol|ergocalciferol|vitamin\s+d3?\s+(?:supplements?|drops|capsules)", ["vitamin_d"]),
    (r"cyanocobalamin|methylcobalamin|hydroxocobalamin|b[\s-]?12\s+(?:injections?|supplements?|shots?)", ["vitamin_b12"]),
    (r"calcium\s+(?:carbonate|citrate|gluconate|supplements?)", ["calcium"]),
    (r"folic\s+acid", ["folate"]),
]

SUPPLEMENT_VERBS = (
    r"supplementation|supplements?|supplemented|supplementing|replacement|replace|replacing|start(?:ed|ing)?|initiat(?:e|ed|ing)|"
    r"prescrib(?:e|ed|ing)|prescription|taking|takes|taken|take|use\s+of|using|uses|treated\s+with|receiving|"
    r"on\s+(?:daily\s+)?(?:supplements?|supplementation)"
)

# ---------------------------------------------------------------------------
# Language scope guard.  NutriCheck interprets ENGLISH only.  These word lists exist
# solely to recognise text that is NOT English so the system can refuse to analyse it.
# ---------------------------------------------------------------------------

EN_WORDS = {
    "the", "and", "of", "with", "was", "were", "is", "are", "has", "have", "had", "no", "not", "patient", "denies",
    "history", "without", "normal", "level", "levels", "report", "test", "results", "for", "to", "in", "on", "at",
    "by", "this", "that", "from", "his", "her", "she", "he", "mother", "father", "low", "high", "deficiency",
    "currently", "present", "evidence", "signs", "reports", "complains", "suspected", "possible", "taking",
    "supplement", "supplements", "serum", "blood", "range", "reference", "result", "laboratory", "follow",
    "visit", "consultation", "return", "days", "days", "year", "years", "age", "male", "female", "diagnosis",
    "treatment", "family", "sister", "brother", "may", "if", "there", "be", "or", "an", "as", "a",
}

# words that are NOT English words (function words / medical terms of other languages)
LANGUAGE_MARKERS: dict[str, set[str]] = {
    "pt": {"de", "da", "das", "dos", "com", "sem", "nao", "para", "paciente", "apresenta", "em", "uma", "por", "foi",
           "esta", "tem", "que", "nos", "nas", "mas", "ou", "mae", "pai", "deficiencia", "carencia", "baixo", "baixa",
           "exame", "resultado", "nega", "historico", "vitamina", "ferro", "ferritina", "calcio", "hemoglobina",
           "acido", "folico", "valor", "niveis", "atualmente", "agora", "consulta", "retorno", "dias", "relatorio",
           "medico", "se", "houver", "investigar", "causa", "tudo", "esta", "todas", "regras", "diga"},
    "es": {"el", "los", "las", "del", "con", "una", "esta", "paciente", "deficiencia", "hierro", "vitamina",
           "niveles", "tiene", "madre", "padre", "nivel", "resultado", "bajo", "baja", "calcio", "ferritina",
           "historia", "antecedentes", "actualmente", "ahora", "por", "para", "pero", "niega"},
    "fr": {"le", "les", "des", "du", "avec", "sans", "pour", "une", "carence", "fer", "vitamine",
           "niveau", "mere", "pere", "taux", "bas", "faible", "nie", "antecedents", "actuellement", "mais"},
    "de": {"der", "das", "mit", "ohne", "und", "ist", "mangel", "eisen", "wert", "niedrig",
           "mutter", "vater", "kein", "keine", "nicht", "aber"},
    "it": {"il", "lo", "gli", "della", "dei", "delle", "con", "senza", "paziente", "carenza", "ferro",
           "vitamina", "livello", "madre", "padre", "basso", "bassa", "ma", "nega", "attualmente"},
}
LANGUAGE_NAMES = {"pt": "Portuguese", "es": "Spanish", "fr": "French", "de": "German", "it": "Italian",
                  "other-script": "a non-Latin script", "unknown": "an unrecognised language"}


@dataclass
class ReferenceTable:
    """Serializable view of the reference ranges (used by GET /v1/nutrients)."""
    rows: list[dict] = field(default_factory=list)


def reference_table() -> list[dict]:
    rows = []
    for a in ANALYTES.values():
        rows.append({
            "analyte": a.key,
            "label": a.label,
            "target": a.target,
            "unit": a.canonical_unit,
            "deficient_below": a.deficient_below,
            "borderline_below": a.borderline_below,
            "high_above": a.high_above,
            "tier": a.tier,
            "weak_evidence": a.weak,
        })
    return rows


_DISPLAY_UNITS = {"ng/ml": "ng/mL", "pg/ml": "pg/mL", "ug/dl": "µg/dL", "ug/l": "µg/L", "mmol/l": "mmol/L", "nmol/l": "nmol/L",
                  "pmol/l": "pmol/L", "g/dl": "g/dL", "g/l": "g/L", "meq/l": "mEq/L", "mg/dl": "mg/dL", "mg/l": "mg/L",
                  "umol/l": "µmol/L", "ng/l": "ng/L", "fl": "fL", "pg": "pg", "%": "%"}


def display_unit(unit: str | None) -> str:
    """'ng/ml' -> 'ng/mL' (units are stored normalised; this is only for people)."""
    return _DISPLAY_UNITS.get(unit or "", unit or "")
