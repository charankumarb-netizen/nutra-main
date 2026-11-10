"""
Privacy helpers: rule-based masking of identifiers for RESEARCH COPIES.

Detects (and replaces by a tag): labelled names, phone numbers, patient / record / national IDs, e-mails and labelled dates of birth.
Laboratory values, units and test dates are left untouched.

HONEST LIMIT: this is pattern-based.  It catches identifiers that are labelled ("Patient: ...", "MRN: ...") or have an obvious shape
(phone, e-mail).  It can miss an unlabelled name or an address.  Review a masked copy before sharing it.
"""

from __future__ import annotations

import re
from typing import Any

_NAME_LABEL = re.compile(
    r"(?P<label>(?i:\b(?:patient(?:\s+name)?|name|referred\s+by|attending(?:\s+physician)?|physician|doctor|dr\.?|mr\.?|mrs\.?|ms\.?|guardian|"
    r"next\s+of\s+kin|signed\s+by)(?![A-Za-z]))[ \t]*[:\-]?[ \t]*)(?P<val>[A-Z][A-Za-z'’\-]+(?:[ \t]+[A-Z][A-Za-z'’\-]+){0,3})")
_ID = re.compile(
    r"(?P<label>\b(?:patient\s*id|pid|mrn|uhid|medical\s+record(?:\s+(?:no|number))?|record\s*(?:no|number)|file\s*(?:no|number)|"
    r"ssn|aadhaar|aadhar|passport(?:\s+no)?|national\s+id|insurance\s*(?:no|id)|ref(?:erence)?\s*(?:no|number)|id)\b\.?\s*[:#\-]?\s*)"
    r"(?P<val>[A-Z0-9][A-Z0-9\-/]{3,})", re.I)
_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_PHONE = re.compile(r"(?<![\w/])\+?\d[\d\s().\-]{7,}\d(?![\w/])")
_AADHAAR = re.compile(r"(?<!\d)\d{4}\s\d{4}\s\d{4}(?!\d)")
_DOB = re.compile(r"(?P<label>\b(?:dob|d\.o\.b\.?|date\s+of\s+birth|born(?:\s+on)?)\b\s*[:\-]?\s*)(?P<val>\d{1,4}[/.\-]\d{1,2}[/.\-]\d{1,4}|\d{1,2}\s+[A-Za-z]{3,9}\.?,?\s+\d{4}|[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{4})", re.I)
_UNIT_AFTER = re.compile(r"\s*(?:ng|pg|ug|mg|g|mmol|nmol|pmol|iu|%|/)", re.I)
_NOT_NAME = {"results", "result", "report", "laboratory", "notes", "history", "plan", "date", "sex", "age", "male", "female", "unknown"}


def mask_pii(text: str) -> dict[str, Any]:
    """-> {"masked": str, "entities": [{"type", "start", "end"}], "counts": {...}}   (offsets refer to the ORIGINAL text)"""
    spans: list[tuple[int, int, str]] = []

    def free(s: int, e: int) -> bool:
        return all(e <= a or s >= b for a, b, _ in spans)

    for rx, tag in ((_DOB, "DOB"), (_ID, "ID"), (_NAME_LABEL, "NAME")):
        for m in rx.finditer(text):
            v0, v1 = m.start("val"), m.end("val")
            val = m.group("val")
            if tag == "NAME" and (val.split()[0].lower() in _NOT_NAME or any(w.lower() in _NOT_NAME for w in val.split()[:1])):
                continue
            if tag == "ID" and not re.search(r"\d", val):
                continue
            if free(v0, v1):
                spans.append((v0, v1, tag))
    for m in _EMAIL.finditer(text):
        if free(m.start(), m.end()):
            spans.append((m.start(), m.end(), "EMAIL"))
    for m in _AADHAAR.finditer(text):
        if free(m.start(), m.end()):
            spans.append((m.start(), m.end(), "ID"))
    for m in _PHONE.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if not (9 <= len(digits) <= 15) or not free(m.start(), m.end()):
            continue
        if re.fullmatch(r"\d{1,4}[/.\-]\d{1,2}[/.\-]\d{1,4}", m.group(0).strip()) or _UNIT_AFTER.match(text[m.end():m.end() + 6] or ""):
            continue
        spans.append((m.start(), m.end(), "PHONE"))
    spans.sort()
    out, pos = [], 0
    for s, e, tag in spans:
        out.append(text[pos:s])
        out.append(f"[{tag}]")
        pos = e
    out.append(text[pos:])
    counts: dict[str, int] = {}
    for _, _, t in spans:
        counts[t] = counts.get(t, 0) + 1
    return {"masked": "".join(out), "entities": [{"type": t, "start": s, "end": e} for s, e, t in spans], "counts": counts,
            "limits": "Pattern-based masking: unlabelled names and addresses can be missed. Review before sharing."}


def mask_structure(obj: Any) -> Any:
    """Mask every string inside a JSON-like structure (used before an analysis is stored)."""
    if isinstance(obj, str):
        return mask_pii(obj)["masked"]
    if isinstance(obj, list):
        return [mask_structure(x) for x in obj]
    if isinstance(obj, dict):
        return {k: mask_structure(v) for k, v in obj.items()}
    return obj


PRIVACY_NOTICE = {
    "where_processed": "Reports are analysed on the machine / server that runs this NutriCheck instance, with rule-based code and (optionally) models "
                       "that run locally. No report text is sent to a third-party service by this code.",
    "what_is_stored": "Nothing is stored unless you give explicit permission. With permission, the extracted results (a masked summary) are stored. "
                      "The original file is stored ONLY if you additionally ask for that, and it is kept in a separate folder from the summaries.",
    "retention": "Stored items are deleted automatically after the retention period (default 90 days, set by the operator) or earlier when you delete them.",
    "your_controls": "You can list, export and delete any stored report, or delete everything, at any time.",
    "security_limits": "Stored files are not encrypted by this software; use disk encryption and protect access to the server. "
                       "Name/ID masking is pattern-based and can miss identifiers.",
    "not_medical_advice": "NutriCheck supports your preparation for a medical appointment; it does not diagnose or recommend supplements.",
}
