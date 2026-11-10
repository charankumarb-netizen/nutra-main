"""Lightweight language detector for NutriCheck."""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class LanguageDetection:
    language: str
    confidence: float


_PT = {
    "paciente", "relatório", "exame", "exames", "resultado",
    "resultados", "hemoglobina", "vitamina", "coleta", "idade",
    "observação", "deficiência", "ferro", "sérico", "ferritina",
    "glicose", "referência", "normal", "anormal", "data",
    "nascimento",
}

_EN = {
    "patient", "report", "medical", "test", "tests", "result",
    "results", "hemoglobin", "vitamin", "collection", "age",
    "notes", "deficiency", "iron", "serum", "ferritin",
    "glucose", "reference", "normal", "abnormal", "date",
    "birth",
}


def detect_language(text: str) -> LanguageDetection:
    value = (text or "").lower()
    words = set(re.findall(r"[a-záàâãéêíóôõúçü]+", value))

    pt = len(words & _PT)
    en = len(words & _EN)

    if pt == 0 and en == 0:
        return LanguageDetection("unknown", 0.0)

    if pt > en:
        return LanguageDetection("pt", min(0.99, 0.60 + 0.08 * (pt - en)))

    if en > pt:
        return LanguageDetection("en", min(0.99, 0.60 + 0.08 * (en - pt)))

    return LanguageDetection("unknown", 0.50)