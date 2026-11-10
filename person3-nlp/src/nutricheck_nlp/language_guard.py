"""
Language-scope guard.

NutriCheck interprets ENGLISH clinical text only.  This module does not translate and does not "support" other
languages: it recognises text that is clearly NOT English so that the pipeline can refuse to interpret it
(a wrong interpretation of a negation or a unit in a language the rules were not written for is worse than none).
"""

from __future__ import annotations

import re

from .lexicon import EN_WORDS, LANGUAGE_MARKERS, LANGUAGE_NAMES

SUPPORTED_LANGUAGE = "en"


def detect_language(norm: str, original: str | None = None) -> dict:
    """-> {"code": "en"|"pt"|"es"|"fr"|"de"|"it"|"other-script"|"unknown", "supported": bool, "name": str, "evidence": {...}}

    `norm` is the normalised (ASCII-folded, lower-case) text; `original` is used only to spot non-Latin scripts.
    "unknown" with English analyte names is treated as English by the caller (short lab-only snippets have no function words).
    """
    original = original if original is not None else norm
    letters = [c for c in original if c.isalpha()]
    non_latin = sum(1 for c in letters if ord(c) > 0x024F)
    if letters and non_latin / len(letters) > 0.20:
        return {"code": "other-script", "supported": False, "name": LANGUAGE_NAMES["other-script"],
                "evidence": {"non_latin_letter_share": round(non_latin / len(letters), 3)}}
    words = re.findall(r"[a-z']+", norm)
    if not words:
        return {"code": "unknown", "supported": True, "name": "no words", "evidence": {}}
    en = sum(w in EN_WORDS for w in words)
    scores = {code: sum(w in markers for w in words) for code, markers in LANGUAGE_MARKERS.items()}
    best_code = max(scores, key=lambda c: scores[c])
    best = scores[best_code]
    if best >= 2 and best > en:
        return {"code": best_code, "supported": False, "name": LANGUAGE_NAMES[best_code],
                "evidence": {"english_words": en, "other_language_words": best}}
    if best >= 1 and en == 0 and len(words) <= 8:
        return {"code": best_code, "supported": False, "name": LANGUAGE_NAMES[best_code],
                "evidence": {"english_words": en, "other_language_words": best}}
    if en >= 1 or best == 0:
        return {"code": "en" if en >= 1 else "unknown", "supported": True, "name": "English", "evidence": {"english_words": en}}
    return {"code": "unknown", "supported": True, "name": "unclear", "evidence": {"english_words": en}}


def unsupported_message(info: dict) -> str:
    return (f"The text appears to be written in {info['name']}. NutriCheck interprets English clinical text only, "
            "so no nutrient assessment was made. Please provide the report in English.")
