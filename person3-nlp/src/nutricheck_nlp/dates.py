"""
Date recognition for English clinical text (and temporal cues such as "currently" / "previously").

Works on the NORMALISED text (lower-case, same length as the original), so offsets can be used on the original.
Numeric dates such as 03/04/2025 are genuinely ambiguous (March 4th or April 3rd); they are NEVER guessed:
the caller may pass date_order="DMY" or "MDY", otherwise the hit is returned with ambiguous=True and iso=None.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field

MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4, "may": 5,
    "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
_MON = r"(?:" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")"

_ISO = re.compile(r"(?<![\d.])(?P<y>(?:19|20)\d{2})-(?P<m>\d{1,2})-(?P<d>\d{1,2})(?![\d])")
_NUMERIC = re.compile(r"(?<![\d.,])(?P<a>\d{1,2})(?P<sep>[/.\-])(?P<b>\d{1,2})(?P=sep)(?P<y>(?:19|20)\d{2}|\d{2})(?![\d])")
_DMY_TEXT = re.compile(r"(?<![\d])(?P<d>\d{1,2})(?:st|nd|rd|th)?[\s\-/]*(?:of\s+)?(?P<mon>" + _MON + r")\.?,?[\s\-/]*(?P<y>(?:19|20)\d{2})(?![\d])")
_MDY_TEXT = re.compile(r"(?<![a-z])(?P<mon>" + _MON + r")\.?\s+(?P<d>\d{1,2})(?:st|nd|rd|th)?,?\s+(?P<y>(?:19|20)\d{2})(?![\d])")
_MY_TEXT = re.compile(r"(?<![a-z])(?P<mon>" + _MON + r")\.?,?\s+(?P<y>(?:19|20)\d{2})(?![\d])")
_YEAR_ONLY = re.compile(r"(?<![\d.,/-])(?:in|during|since|from|of|dated|year|back\s+in|as\s+of)\s+(?P<y>(?:19|20)\d{2})(?![\d/-]|[.,]\d|\s*(?:iu|mg|ng|pg|ug|mcg|units?)\b)")

_DOC_DATE_LABEL = re.compile(
    r"(?:collect\w*|sampl\w*|specimen|report(?:ed)?|date|visit|encounter|drawn|draw|exam(?:ination)?|tested|test\s+date|dated)[^\n\d]{0,24}$")


@dataclass
class DateHit:
    start: int
    end: int
    raw: str
    iso: str | None                  # YYYY-MM-DD | YYYY-MM | YYYY | None when ambiguous / invalid
    precision: str                   # day | month | year
    ambiguous: bool = False
    candidates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"raw": self.raw, "iso": self.iso, "precision": self.precision,
                "ambiguous": self.ambiguous, "candidates": self.candidates}


def _valid(y: int, m: int, d: int) -> bool:
    return 1900 <= y <= 2100 and 1 <= m <= 12 and 1 <= d <= calendar.monthrange(y, m)[1]


def _year(y: str) -> int:
    n = int(y)
    return n if n >= 100 else (2000 + n if n <= 69 else 1900 + n)


def find_dates(norm: str, date_order: str | None = None) -> list[DateHit]:
    """All dates in `norm`, sorted by position. date_order: None | "DMY" | "MDY" (used only for ambiguous numeric dates)."""
    hits: list[DateHit] = []

    def overlaps(s: int, e: int) -> bool:
        return any(not (e <= h.start or s >= h.end) for h in hits)

    for m in _ISO.finditer(norm):
        y, mo, d = int(m["y"]), int(m["m"]), int(m["d"])
        if _valid(y, mo, d):
            hits.append(DateHit(m.start(), m.end(), m.group(0), f"{y:04d}-{mo:02d}-{d:02d}", "day"))
    for m in _DMY_TEXT.finditer(norm):
        y, mo, d = int(m["y"]), MONTHS[m["mon"]], int(m["d"])
        if _valid(y, mo, d) and not overlaps(m.start(), m.end()):
            hits.append(DateHit(m.start(), m.end(), m.group(0), f"{y:04d}-{mo:02d}-{d:02d}", "day"))
    for m in _MDY_TEXT.finditer(norm):
        y, mo, d = int(m["y"]), MONTHS[m["mon"]], int(m["d"])
        if _valid(y, mo, d) and not overlaps(m.start(), m.end()):
            hits.append(DateHit(m.start(), m.end(), m.group(0), f"{y:04d}-{mo:02d}-{d:02d}", "day"))
    for m in _NUMERIC.finditer(norm):
        if overlaps(m.start(), m.end()):
            continue
        a, b, y = int(m["a"]), int(m["b"]), _year(m["y"])
        dmy, mdy = _valid(y, b, a), _valid(y, a, b)
        raw = m.group(0)
        if not dmy and not mdy:
            continue
        cand_dmy, cand_mdy = f"{y:04d}-{b:02d}-{a:02d}", f"{y:04d}-{a:02d}-{b:02d}"
        if dmy and mdy and a != b:
            if date_order == "DMY":
                hits.append(DateHit(m.start(), m.end(), raw, cand_dmy, "day"))
            elif date_order == "MDY":
                hits.append(DateHit(m.start(), m.end(), raw, cand_mdy, "day"))
            else:
                hits.append(DateHit(m.start(), m.end(), raw, None, "day", True, [cand_dmy, cand_mdy]))
        else:
            hits.append(DateHit(m.start(), m.end(), raw, cand_dmy if dmy else cand_mdy, "day"))
    for m in _MY_TEXT.finditer(norm):
        y, mo = int(m["y"]), MONTHS[m["mon"]]
        if 1900 <= y <= 2100 and not overlaps(m.start(), m.end()):
            hits.append(DateHit(m.start(), m.end(), m.group(0), f"{y:04d}-{mo:02d}", "month"))
    for m in _YEAR_ONLY.finditer(norm):
        y = int(m["y"])
        if 1900 <= y <= 2100 and not overlaps(m.start(), m.end()):
            hits.append(DateHit(m.start(), m.end(), m.group(0), f"{y:04d}", "year"))
    hits.sort(key=lambda h: h.start)
    return hits[:500]


def document_date(norm: str, hits: list[DateHit]) -> DateHit | None:
    """The collection / report date of the document (a date preceded on its line by a label such as 'Collected:')."""
    for h in hits:
        if h.precision == "year":
            continue
        line_start = norm.rfind("\n", 0, h.start) + 1
        if _DOC_DATE_LABEL.search(norm[line_start:h.start]):
            return h
    return None


def date_for_position(norm: str, hits: list[DateHit], start: int, end: int, doc_date: DateHit | None) -> tuple[DateHit | None, str | None]:
    """Pick the date that belongs to a value located at [start, end): same line, then same sentence, then the document date."""
    line_s = norm.rfind("\n", 0, start) + 1
    p1 = norm.rfind(". ", 0, start)
    sent_s = max(p1 + 2 if p1 >= 0 else 0, norm.rfind(";", 0, start) + 1, line_s)
    m = re.search(r"[.;!](?=\s|$)|\n", norm[end:])
    sent_e = end + m.start() if m else len(norm)
    in_sent = [h for h in hits if h.start >= sent_s and h.end <= sent_e and h is not doc_date]
    if in_sent:
        return min(in_sent, key=lambda h: min(abs(h.start - end), abs(start - h.end))), "sentence"
    if doc_date is not None:
        return doc_date, "document"
    return None, None


_CURRENT = re.compile(r"(?<![a-z])(?:currently|now|today|at\s+present|presently|latest|most\s+recent|current|this\s+visit|as\s+of\s+today|nowadays|at\s+this\s+visit)(?![a-z])")
_PAST = re.compile(r"(?<![a-z])(?:previously|formerly|prior|earlier|in\s+the\s+past|last\s+(?:year|month|week)|(?:years?|months?|weeks?)\s+ago|baseline|initially|before|previous|old|past)(?![a-z])")


def temporal_cue(norm: str, name_start: int, value_end: int) -> str | None:
    """'current' | 'past' | None for the value at [name_start, value_end), using the words right around it (same sentence)."""
    p1 = norm.rfind(". ", 0, name_start)
    sent_s = max(p1 + 2 if p1 >= 0 else 0, norm.rfind(";", 0, name_start) + 1, norm.rfind("\n", 0, name_start) + 1, 0)
    m = re.search(r"[.;!](?=\s|$)|\n", norm[value_end:])
    sent_e = value_end + m.start() if m else len(norm)
    before = norm[max(sent_s, name_start - 45):name_start]
    after = norm[value_end:min(sent_e, value_end + 25)]
    cands: list[tuple[int, str]] = []
    for rx, label in ((_CURRENT, "current"), (_PAST, "past")):
        cands += [(len(before) - mm.end(), label) for mm in rx.finditer(before)]
        cands += [(mm.start() + 1, label) for mm in rx.finditer(after)]
    return min(cands)[1] if cands else None
