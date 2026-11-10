"""
NutriCheck Multilingual Medical Knowledge & Terminology Dictionary.

Provides:
- Language detection function words
- Multilingual biomarker terminology and regex compiler
- Number parsing supporting comma decimals (e.g. 10,8 / 13,5) and dot decimals (10.8)
- Additional CBC / Hematology markers: MCV / VCM, MCH / HCM, MCHC / CHCM, Hematocrit
- Reference interval markers (VR, Valores de referência, etc.)
- Separators for key-value, lists, tables, dotted lines, and paragraphs
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Supported Languages
# ---------------------------------------------------------------------------

SUPPORTED_LANGUAGES: dict[str, str] = {
    "en": "English",
    "pt": "Portuguese (Português)",
    "es": "Spanish (Español)",
    "fr": "French (Français)",
    "de": "German (Deutsch)",
    "hi": "Hindi (हिन्दी)",
}

# Frequent words used for language identification
LANGUAGE_FUNCTION_WORDS: dict[str, set[str]] = {
    "pt": {
        "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas", "para", "com",
        "por", "sem", "ou", "se", "como", "mais", "foi", "sao", "estao", "paciente",
        "exame", "resultado", "resultados", "referencia", "valores", "sangue", "laudo",
        "data", "medico", "solicitante", "apresentou", "observado", "dosagem", "idade",
        "anos", "sexo", "masculino", "feminino", "sintomas", "clinica", "observacao",
        "laboratoriais", "relacionados", "neurologicos", "fraqueza", "memoria"
    },
    "es": {
        "de", "del", "la", "las", "el", "los", "en", "para", "con", "por", "sin", "o",
        "si", "como", "mas", "fue", "son", "estan", "paciente", "analisis", "resultado",
        "resultados", "referencia", "valores", "sangre", "informe", "fecha", "medico",
        "presento", "observado", "determinacion", "edad", "anos", "sexo", "sintomas"
    },
    "fr": {
        "de", "du", "des", "la", "le", "les", "en", "dans", "pour", "avec", "par", "sans",
        "ou", "si", "comme", "plus", "ete", "sont", "patient", "analyse", "resultat",
        "resultats", "reference", "valeurs", "sang", "prelevement", "date", "medecin",
        "dosage", "taux", "normes", "age", "ans", "sexe"
    },
    "de": {
        "der", "die", "das", "den", "dem", "des", "und", "in", "von", "mit", "fur", "fuer",
        "auf", "ist", "sind", "war", "nicht", "ohne", "patient", "befund", "ergebnis",
        "ergebnisse", "referenz", "referenzbereich", "blut", "datum", "arzt", "labor",
        "messung", "wert", "werte", "untersuchung", "alter", "jahre", "geschlecht"
    },
}


# ---------------------------------------------------------------------------
# Biomarker Terminology
# ---------------------------------------------------------------------------

@dataclass
class BiomarkerTerminology:
    key: str
    target: str
    canonical_name: str
    canonical_unit: str
    synonyms_by_lang: dict[str, list[str]] = field(default_factory=dict)
    abbreviations: list[str] = field(default_factory=list)
    default_unit: str = ""

    def all_synonyms(self) -> list[str]:
        items: list[str] = []
        for syn_list in self.synonyms_by_lang.values():
            items.extend(syn_list)
        items.extend(self.abbreviations)
        # Sort descending by length to avoid sub-string collisions
        items = list(dict.fromkeys(items))
        items.sort(key=lambda s: -len(s))
        return items

    def compile_regex_pattern(self) -> str:
        escaped = []
        for s in self.all_synonyms():
            # Allow flexible spaces, dots, and hyphens in biomarker names
            sub = re.escape(s.lower())
            sub = sub.replace(r"\ ", r"[\s\.\-_]+")
            sub = sub.replace(r"\-", r"[\s\.\-_]+")
            escaped.append(sub)
        return r"(?:" + "|".join(escaped) + r")"


MULTILINGUAL_BIOMARKERS: dict[str, BiomarkerTerminology] = {
    # ------------------ VITAMIN B12 ------------------
    "vitamin_b12": BiomarkerTerminology(
        key="vitamin_b12",
        target="vitamin_b12",
        canonical_name="Vitamin B12",
        canonical_unit="pg/ml",
        synonyms_by_lang={
            "en": [
                "vitamin b12", "vitamin b-12", "vitamin b 12", "vit b12", "vit-b12", "vit b-12",
                "cobalamin", "cyanocobalamin", "serum b12", "total vitamin b12", "b12 level"
            ],
            "pt": [
                "vitamina b12", "vitamina b-12", "vitamina b 12", "vit b12", "vit-b12",
                "cobalamina", "cianocobalamina", "cobalamina serica", "dosagem de vitamina b12",
                "b12 serica", "nivel de vitamina b12"
            ],
            "es": [
                "vitamina b12", "vitamina b-12", "vitamina b 12", "vit b12", "vit-b12",
                "cobalamina", "cianocobalamina", "cobalamina serica", "b12 serica"
            ],
            "fr": [
                "vitamine b12", "vitamine b-12", "vitamine b 12", "vit b12", "vit-b12",
                "cobalamine", "cyanocobalamine", "taux de vitamine b12"
            ],
            "de": [
                "vitamin b12", "vitamin b-12", "vitamin b 12", "vit b12", "vit-b12",
                "cobalamin", "zyanokobalamin", "vitamin-b12-spiegel"
            ],
            "hi": [
                "विटामिन बी12", "विटामिन बी-12", "विटामिन b12", "विटामिन b-12"
            ],
        },
        abbreviations=["b12", "vit b12", "vit-b12", "b-12", "b 12"],
        default_unit="pg/ml",
    ),

    # ------------------ VITAMIN D ------------------
    "vitamin_d": BiomarkerTerminology(
        key="vitamin_d",
        target="vitamin_d",
        canonical_name="Vitamin D",
        canonical_unit="ng/ml",
        synonyms_by_lang={
            "en": [
                "vitamin d", "vitamin d3", "25-hydroxyvitamin d", "25-oh vitamin d", "25-oh-d",
                "25-hydroxy vitamin d", "calcidiol", "total vitamin d", "cholecalciferol",
                "25-oh calcidiol", "vitamin d total", "vit d"
            ],
            "pt": [
                "vitamina d", "vitamina d3", "vitamina d-3", "25-hidroxivitamina d", "25-oh vitamina d",
                "25-oh-vitamina d", "calcidiol", "vitamina d total", "vit d", "25-hidroxi-vitamina d",
                "dosagem de vitamina d", "25 hidroxivitamina d", "25 oh vitamina d"
            ],
            "es": [
                "vitamina d", "vitamina d3", "25-hidroxivitamina d", "25-oh vitamina d",
                "calcidiol", "vitamina d total", "vit d", "25-hidroxicolecalciferol"
            ],
            "fr": [
                "vitamine d", "vitamine d3", "25-hydroxyvitamine d", "25-oh vitamine d",
                "calcidiol", "vitamine d totale", "vit d"
            ],
            "de": [
                "vitamin d", "vitamin d3", "25-hydroxyvitamin d", "25-oh vitamin d",
                "calcidiol", "vitamin d total", "vit d"
            ],
            "hi": [
                "विटामिन डी", "विटामिन d", "25-हाइड्रॉक्सी विटामिन डी"
            ],
        },
        abbreviations=["25-oh-d", "25-oh d", "25ohd", "vit d", "vit-d", "vit d3"],
        default_unit="ng/ml",
    ),

    # ------------------ IRON & FERRITIN ------------------
    "ferritin": BiomarkerTerminology(
        key="ferritin",
        target="iron",
        canonical_name="Ferritin",
        canonical_unit="ng/ml",
        synonyms_by_lang={
            "en": ["ferritin", "serum ferritin", "plasma ferritin", "total ferritin"],
            "pt": ["ferritina", "ferritina serica", "dosagem de ferritina", "ferritina no soro"],
            "es": ["ferritina", "ferritina serica", "ferritina en suero"],
            "fr": ["ferritine", "ferritine serique", "dosage de la ferritine"],
            "de": ["ferritin", "serumferritin"],
            "hi": ["फेरिटिन", "सीरम फेरिटिन"],
        },
        abbreviations=["fer"],
        default_unit="ng/ml",
    ),
    "serum_iron": BiomarkerTerminology(
        key="serum_iron",
        target="iron",
        canonical_name="Serum Iron",
        canonical_unit="ug/dl",
        synonyms_by_lang={
            "en": ["serum iron", "plasma iron", "iron", "fe", "total iron"],
            "pt": ["ferro serico", "ferro no soro", "ferro total", "dosagem de ferro", "ferro"],
            "es": ["hierro serico", "hierro total", "hierro en suero", "hierro"],
            "fr": ["fer serique", "fer total", "fer"],
            "de": ["serumeisen", "eisen"],
            "hi": ["सीरम आयरन", "आयरन"],
        },
        abbreviations=["fe", "fe++"],
        default_unit="ug/dl",
    ),
    "tsat": BiomarkerTerminology(
        key="tsat",
        target="iron",
        canonical_name="Transferrin Saturation",
        canonical_unit="%",
        synonyms_by_lang={
            "en": ["transferrin saturation", "tsat", "iron saturation index", "saturation of transferrin"],
            "pt": ["saturacao de transferrina", "indice de saturacao de transferrina", "saturacao da transferrina"],
            "es": ["saturacion de transferrina", "indice de saturacion de transferrina"],
            "fr": ["saturation de la transferrine", "coefficient de saturation de la transferrine"],
            "de": ["transferrinsattigung", "transferrinsaettigung"],
            "hi": ["ट्रांसफरिन संतृप्ति"],
        },
        abbreviations=["tsat", "ist", "csf"],
        default_unit="%",
    ),

    # ------------------ CALCIUM ------------------
    "calcium_total": BiomarkerTerminology(
        key="calcium_total",
        target="calcium",
        canonical_name="Total Calcium",
        canonical_unit="mg/dl",
        synonyms_by_lang={
            "en": ["total calcium", "serum calcium", "calcium", "blood calcium", "calcium total"],
            "pt": ["calcio total", "calcio serico", "calcio", "calcemia", "calcio no soro", "dosagem de calcio"],
            "es": ["calcio total", "calcio serico", "calcio", "calcemia"],
            "fr": ["calcium total", "calcium serique", "calcium", "calcemie"],
            "de": ["gesamtkalzium", "serumkalzium", "kalzium", "calcium"],
            "hi": ["कैल्शियम", "टोटल कैल्शियम"],
        },
        abbreviations=["ca", "ca-t"],
        default_unit="mg/dl",
    ),
    "calcium_ionized": BiomarkerTerminology(
        key="calcium_ionized",
        target="calcium",
        canonical_name="Ionized Calcium",
        canonical_unit="mmol/l",
        synonyms_by_lang={
            "en": ["ionized calcium", "ionised calcium", "free calcium"],
            "pt": ["calcio ionico", "calcio ionizado", "calcio livre"],
            "es": ["calcio ionico", "calcio ionizado", "calcio libre"],
            "fr": ["calcium ionise", "calcium libre"],
            "de": ["ionisiertes kalzium", "ionisiertes calcium"],
            "hi": ["आयनीकृत कैल्शियम"],
        },
        abbreviations=["ca++", "ica"],
        default_unit="mmol/l",
    ),
    "calcium_corrected": BiomarkerTerminology(
        key="calcium_corrected",
        target="calcium",
        canonical_name="Corrected Calcium",
        canonical_unit="mg/dl",
        synonyms_by_lang={
            "en": ["corrected calcium", "albumin-corrected calcium"],
            "pt": ["calcio corrigido", "calcio corrigido pela albumina"],
            "es": ["calcio corregido", "calcio corregido por albumina"],
            "fr": ["calcium corrige"],
            "de": ["korrigiertes kalzium"],
            "hi": ["करेक्टेड कैल्शियम"],
        },
        abbreviations=["ca-corr"],
        default_unit="mg/dl",
    ),

    # ------------------ FOLATE ------------------
    "folate": BiomarkerTerminology(
        key="folate",
        target="folate",
        canonical_name="Folate",
        canonical_unit="ng/ml",
        synonyms_by_lang={
            "en": ["folate", "folic acid", "serum folate", "vitamin b9", "plasma folate"],
            "pt": ["folato", "acido folico", "folato serico", "vitamina b9", "vitamina b-9", "dosagem de folato", "acido folico serico"],
            "es": ["folato", "acido folico", "folato serico", "vitamina b9"],
            "fr": ["folate", "acide folique", "folates seriques", "vitamine b9"],
            "de": ["folat", "folsaure", "folsäure", "serumfolat", "vitamin b9"],
            "hi": ["फोलेट", "फोलिक एसिड"],
        },
        abbreviations=["fol", "b9", "vit b9"],
        default_unit="ng/ml",
    ),

    # ------------------ HEMATOLOGY & CBC ------------------
    "hemoglobin": BiomarkerTerminology(
        key="hemoglobin",
        target="anemia",
        canonical_name="Hemoglobin",
        canonical_unit="g/dl",
        synonyms_by_lang={
            "en": ["hemoglobin", "haemoglobin", "blood hemoglobin", "total hemoglobin", "hgb level"],
            "pt": ["hemoglobina", "taxa de hemoglobina", "dosagem de hemoglobina", "hb sanguinea", "hemoglobina total"],
            "es": ["hemoglobina", "tasa de hemoglobina", "hemoglobina total"],
            "fr": ["hemoglobine", "taux d'hemoglobine", "hemoglobine totale"],
            "de": ["hamoglobin", "hämoglobin", "haemoglobin"],
            "hi": ["हीमोग्लोबिन"],
        },
        abbreviations=["hb", "hgb"],
        default_unit="g/dl",
    ),

    "hematocrit": BiomarkerTerminology(
        key="hematocrit",
        target="anemia",
        canonical_name="Hematocrit",
        canonical_unit="%",
        synonyms_by_lang={
            "en": ["hematocrit", "haematocrit", "packed cell volume", "pcv"],
            "pt": ["hematocrito", "volume globular"],
            "es": ["hematocrito", "volumen globular"],
            "fr": ["hematocrite", "volume globulaire"],
            "de": ["hamatokrit", "hämatokrit"],
            "hi": ["हेमाटोक्रिट"],
        },
        abbreviations=["ht", "hct", "pcv"],
        default_unit="%",
    ),

    # MCV / VCM (Mean Corpuscular Volume / Volume Corpuscular Médio)
    "mcv": BiomarkerTerminology(
        key="mcv",
        target="anemia",
        canonical_name="Mean Corpuscular Volume (MCV)",
        canonical_unit="fl",
        synonyms_by_lang={
            "en": ["mcv", "mean corpuscular volume", "mean cell volume", "erythrocyte mean volume"],
            "pt": ["vcm", "volume corpuscular medio", "volume globular medio", "vgm", "volume eritrocitario medio"],
            "es": ["vcm", "volumen corpuscular medio", "volumen globular medio"],
            "fr": ["vgm", "volume globulaire moyen", "volume corpusculaire moyen"],
            "de": ["mcv", "mittleres korpuskularvolumen", "mittleres zellvolumen"],
            "hi": ["एमसीवी", "माध्य कणिका आयतन"],
        },
        abbreviations=["vcm", "mcv", "vgm"],
        default_unit="fl",
    ),

    # MCH / HCM (Mean Corpuscular Hemoglobin / Hemoglobina Corpuscular Média)
    "mch": BiomarkerTerminology(
        key="mch",
        target="anemia",
        canonical_name="Mean Corpuscular Hemoglobin (MCH)",
        canonical_unit="pg",
        synonyms_by_lang={
            "en": ["mch", "mean corpuscular hemoglobin", "mean cell hemoglobin"],
            "pt": ["hcm", "hemoglobina corpuscular media", "hgm", "hemoglobina globular media"],
            "es": ["hcm", "hemoglobina corpuscular media"],
            "fr": ["tcmh", "teneur corpusculaire moyenne en hemoglobine"],
            "de": ["mch", "mittleres korpuskularhamoglobin"],
            "hi": ["एमसीएच"],
        },
        abbreviations=["hcm", "mch", "tcmh"],
        default_unit="pg",
    ),

    # MCHC / CHCM (Mean Corpuscular Hemoglobin Concentration / Concentração de Hemoglobina Corpuscular Média)
    "mchc": BiomarkerTerminology(
        key="mchc",
        target="anemia",
        canonical_name="Mean Corpuscular Hemoglobin Concentration (MCHC)",
        canonical_unit="g/dl",
        synonyms_by_lang={
            "en": ["mchc", "mean corpuscular hemoglobin concentration", "mean cell hemoglobin concentration"],
            "pt": ["chcm", "concentracao de hemoglobina corpuscular media", "concentracao corpuscular media de hemoglobina"],
            "es": ["chcm", "concentracion de hemoglobina corpuscular media"],
            "fr": ["ccmh", "concentration corpusculaire moyenne en hemoglobine"],
            "de": ["mchc", "mittlere korpuskulare hamoglobinkonzentration"],
            "hi": ["एमसीएचसी"],
        },
        abbreviations=["chcm", "mchc", "ccmh"],
        default_unit="g/dl",
    ),
}


# ---------------------------------------------------------------------------
# Separators and Reference Interval Keys
# ---------------------------------------------------------------------------

MULTILINGUAL_SEPARATORS_PATTERN = (
    r"(?:"
    r"[ \t]*[:=][ \t]*"                          # Colon or equals
    r"|[ \t]*[-–—][ \t]*(?!\d)"                  # Dash separator
    r"|[ \t]+(?:de|do|da|em|no|na|para|dans|von|of|at|was|is|foi|sao|estao|nivel|level|resultado|result|valor|value)[ \t]+"
    r"|[ \t]*\.{2,}[ \t]*"                        # Dotted lines like .......... 100
    r"|[ \t]{2,}"                                # Multi-space or tabular columns
    r"|\t+"                                      # Tab character
    r"|[ \t]+"                                   # Single space
    r")"
)

MULTILINGUAL_REF_KEYS = (
    r"valores?\s+de\s+referencia|valor\s+de\s+referencia|valores\s+de\s+ref|"
    r"intervalo\s+de\s+referencia|valores\s+normais|faixa\s+de\s+referencia|"
    r"vr|ref\b|reference\s+range|reference\s+interval|normal\s+range|reference|"
    r"valeurs\s+de\s+reference|normes|referenzbereich|normalwert|referenz"
)

MULTILINGUAL_ORDERED_TEST_CUES = (
    r"solicito|solicitado|pedir|pedido|prescrito|indico|programar|jejum|"
    r"rotina|orientacoes|a\s+realizar|para\s+realizar|ordered|prescribed|order"
)


# ---------------------------------------------------------------------------
# Number Parser
# ---------------------------------------------------------------------------

def parse_multilingual_number(raw_text: str) -> float:
    """
    Parses a clinical number accurately in either English or continental European format.
    Handles:
      - '10,8'   -> 10.8 (Portuguese / European decimal comma)
      - '13,5'   -> 13.5 (Portuguese / European decimal comma)
      - '10.8'   -> 10.8 (English decimal point)
      - '100'    -> 100.0 (Integer)
      - '1.200'  -> 1200.0 (Thousands dot)
      - '1,200.5'-> 1200.5 (Thousands comma + decimal dot)
      - '1.200,5'-> 1200.5 (Thousands dot + decimal comma)
    """
    cleaned = raw_text.strip().replace(" ", "")
    if not cleaned:
        raise ValueError("Empty number string")

    # Mixed separators: both comma and dot
    if "," in cleaned and "." in cleaned:
        last_comma = cleaned.rfind(",")
        last_dot = cleaned.rfind(".")
        if last_comma > last_dot:
            # e.g. 1.200,5 -> dot is thousands, comma is decimal
            num_str = cleaned.replace(".", "").replace(",", ".")
            return float(num_str)
        else:
            # e.g. 1,200.5 -> comma is thousands, dot is decimal
            num_str = cleaned.replace(",", "")
            return float(num_str)

    # Only comma present
    if "," in cleaned:
        parts = cleaned.split(",")
        if len(parts) > 2:
            # Multiple commas: e.g. 1,000,000
            return float(cleaned.replace(",", ""))
        if len(parts) == 2:
            # e.g. 10,8 or 13,5 -> decimal comma
            return float(f"{parts[0]}.{parts[1]}")

    # Only dot present
    if "." in cleaned:
        parts = cleaned.split(".")
        if len(parts) > 2:
            return float(cleaned.replace(".", ""))
        if len(parts) == 2:
            # If 3 digits and integer >= 10, e.g. 1.200 -> 1200
            if len(parts[1]) == 3 and int(parts[0]) >= 10:
                return float(cleaned.replace(".", ""))
            return float(cleaned)

    # Standard integer
    return float(cleaned)

