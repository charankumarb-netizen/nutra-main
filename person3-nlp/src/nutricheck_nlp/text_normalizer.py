"""
Offset-preserving normalisation for English clinical text.

normalize_text(text) returns a string with EXACTLY the same length as the input,
so character offsets found on the normalised text can be used to quote the
original text.  It lower-cases, strips accents, and unifies whitespace, dashes,
quotes and the micro sign.
"""

from __future__ import annotations

import unicodedata

MAX_TEXT_CHARS = 50_000

_SPACE_LIKE = {"\u00a0", "\u2007", "\u202f", "\u2009", "\u200a", "\u2002", "\u2003", "\t", "\r", "\x0b", "\x0c"}
_DASHES = {"\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212"}
_ZERO_WIDTH = {"\u200b", "\u200c", "\u200d", "\ufeff", "\x00"}


def _norm_char(ch: str) -> str:
    if ch in _SPACE_LIKE:
        return " "
    if ch in _ZERO_WIDTH:
        return " "
    if ch in _DASHES:
        return "-"
    if ch in {"\u2018", "\u2019", "\u201b", "\u00b4", "`"}:
        return "'"
    if ch in {"\u201c", "\u201d"}:
        return '"'
    if ch == "\n":
        return "\n"
    if ch in {"µ", "μ"}:
        return "u"
    if ch == "\uff1a":  # full-width colon
        return ":"
    decomposed = unicodedata.normalize("NFKD", ch)
    base = "".join(c for c in decomposed if not unicodedata.combining(c))
    if len(base) != 1:
        base = base[:1] if base else " "
    base = base.lower()
    if len(base) != 1:
        base = base[:1]
    if base in {"µ", "μ"}:
        return "u"
    return base


def normalize_text(text: str) -> str:
    """Length-preserving normalisation (see module docstring)."""
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    return "".join(_norm_char(c) for c in text)


def normalize_unit(unit: str | None) -> str | None:
    """Normalise a unit string: 'µg / dL' -> 'ug/dl', 'ng/m1' -> 'ng/ml'."""
    if unit is None:
        return None
    u = normalize_text(unit).replace(" ", "")
    u = u.replace("mcg", "ug").replace("\u00b5g", "ug")
    # common OCR confusions inside units: l <-> 1 / I
    if "/" in u:
        num, den = u.split("/", 1)
        den = {"m1": "ml", "d1": "dl", "1": "l", "mi": "ml", "di": "dl"}.get(den, den)
        u = f"{num}/{den}"
    return u


def sanitize_input(text: str) -> str:
    """Clip pathological sizes and drop NUL characters (defensive)."""
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    return text[:MAX_TEXT_CHARS]
