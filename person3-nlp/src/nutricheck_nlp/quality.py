"""
"AI that knows when NOT to predict": document-quality and reliability checks.

"I cannot reliably determine this from the available information" is a SUCCESSFUL, SAFE outcome here, not a failure.
Checks (each one is reported with ok / needs-attention and a plain-language action for the user):
    1 text readable?            2 values extracted correctly?      3 units present and unambiguous?
    4 language supported?       5 enough relevant facts?
"""

from __future__ import annotations

from typing import Any

OCR_MIN_CONFIDENCE = 0.85
OCR_LINE_MIN_CONFIDENCE = 0.90

_REASON_TEXT = {
    "ambiguous_ocr_characters": "contains characters that OCR often mixes up with digits",
    "ambiguous_decimal_format": "uses a comma that may be a decimal separator",
    "implausible_value": "is physiologically implausible",
    "unit_missing": "has no unit",
    "ambiguous_date": "has an ambiguous date",
    "comparator_not_decisive": "is a '<' or '>' value that cannot settle the question",
    "negative_value": "is negative",
    "scientific_notation_or_non_finite": "is not a valid finite number",
    "unreadable_number": "could not be read as a number",
    "unparseable_number": "could not be read as a number",
    "low_ocr_confidence": "was read by OCR with low confidence",
    "too_many_values": "is one of too many repeated values",
}


def reason_text(code: str) -> str:
    if code.startswith("unit_mismatch"):
        return "has a unit that does not belong to this test"
    return _REASON_TEXT.get(code, code.replace("_", " "))


def garbage_ratio(text: str) -> float:
    """Share of characters that are neither letters, digits, whitespace nor common clinical punctuation."""
    if not text:
        return 0.0
    ok = set(".,:;()[]/%-+<>=*#'\"°µ·&?!_|~")
    bad = sum(1 for c in text if not (c.isalnum() or c.isspace() or c in ok))
    return bad / len(text)


def assess_quality(*, text: str, language: dict, labs: list[dict], rejected: list[dict], mentions: list[dict],
                   ocr_confidence: float | None, low_conf_lines: list[dict], page_report: list[dict] | None,
                   nutrients: list[dict]) -> dict[str, Any]:
    checks: list[dict] = []
    actions: list[str] = []
    verdict = "ok"

    # 1 readable text -------------------------------------------------------------------------
    readable, detail = True, "The text layer / OCR output looks readable."
    if not text.strip():
        readable, detail = False, "No text could be read."
    elif garbage_ratio(text) > 0.15:
        readable, detail = False, "The text contains a large share of unreadable symbols."
    elif ocr_confidence is not None and ocr_confidence < OCR_MIN_CONFIDENCE:
        readable, detail = False, f"OCR confidence is low ({ocr_confidence:.2f})."
    elif low_conf_lines:
        detail = f"{len(low_conf_lines)} line(s) were read with low OCR confidence."
    if not readable:
        actions.append("Upload a clearer image or a text-based PDF of the report.")
    checks.append({"id": "text_readable", "question": "Is the scanned text readable?", "ok": readable, "detail": detail})

    # 2 values extracted correctly -------------------------------------------------------------
    bad_labs = [l for l in labs if l["verify_reasons"] and ("unit_missing" not in l["verify_reasons"] or len(l["verify_reasons"]) > 1)]
    invalid = [r for r in rejected if r["reason"] not in ("date_or_reference_range",)]
    lowc = [l for l in labs if "low_ocr_confidence" in l["verify_reasons"]]
    values_ok = not bad_labs and not invalid
    vdetail = "Every extracted value passed the sanity checks."
    if not values_ok:
        items = [f"“{l['raw_text']}” {reason_text(l['verify_reasons'][0])}" for l in bad_labs] + \
                [f"“{r['raw_text']}” {reason_text(r['reason'])}" for r in invalid]
        vdetail = "; ".join(items)
        for l in bad_labs:
            actions.append(f"Please verify the value of {l['label']} (page {l['location'].get('page') or 'n/a'}, line {l['location']['line']}) "
                           f"against the original: it {reason_text(l['verify_reasons'][0])}.")
        for r in invalid:
            actions.append(f"Please check “{r['raw_text']}” in the original report: it {reason_text(r['reason'])}.")
    elif lowc:
        values_ok, vdetail = False, f"{len(lowc)} value(s) came from low-confidence OCR lines."
    for l in lowc:
        actions.append(f"Please verify {l['label']} (page {l['location'].get('page') or 'n/a'}, line {l['location']['line']}): the OCR confidence for that line is low.")
    checks.append({"id": "values_extracted", "question": "Were the lab values extracted correctly?", "ok": values_ok, "detail": vdetail})

    # 3 units -----------------------------------------------------------------------------------
    no_unit = [l for l in labs if l["assumed_unit"]]
    unit_bad = [r for r in rejected if r["reason"].startswith("unit_mismatch")]
    units_ok = not no_unit and not unit_bad
    udetail = "Every value has a recognised unit."
    if not units_ok:
        udetail = "; ".join([f"{l['label']} has no unit" for l in no_unit] + [f"“{r['raw_text']}” has an unexpected unit" for r in unit_bad])
        for l in no_unit:
            actions.append(f"Confirm the unit for {l['label']} ({l['value']:g}): it was missing, so the result is provisional.")
    checks.append({"id": "units_clear", "question": "Is any unit missing or ambiguous?", "ok": units_ok, "detail": udetail})

    # 4 language --------------------------------------------------------------------------------
    lang_ok = bool(language.get("supported", True))
    if not lang_ok:
        actions.append("Provide the report in English; this system does not interpret other languages.")
    checks.append({"id": "language_supported", "question": "Is the report in a supported language?", "ok": lang_ok,
                   "detail": "English." if lang_ok else f"The text appears to be {language.get('name')}."})

    # 5 enough facts ----------------------------------------------------------------------------
    usable = [l for l in labs if l["usable"] and not l["assumed_unit"]]
    stated = [m for m in mentions if m["stance"] != "ignored"]
    enough = bool(usable or stated or bad_labs or invalid or no_unit)
    if not enough and lang_ok and text.strip():
        actions.append("No nutrient values or statements were found. Upload the page that contains the laboratory results.")
    checks.append({"id": "enough_facts", "question": "Are there enough relevant facts to support a finding?", "ok": enough,
                   "detail": f"{len(usable)} usable lab value(s) and {len(stated)} explicit statement(s) found." if enough
                   else "No usable lab value or explicit statement about the target nutrients was found."})

    # verdict -----------------------------------------------------------------------------------
    if not lang_ok or not text.strip() or (ocr_confidence is not None and ocr_confidence < 0.6):
        verdict = "cannot_determine"
    elif not enough:
        verdict = "insufficient_information"
    elif not all(c["ok"] for c in checks):
        verdict = "verify"
    abstained = [n["key"] for n in nutrients if n.get("reliability") == "abstain"]
    if verdict == "ok" and abstained:
        verdict = "verify"

    # 4-tier category classification ------------------------------------------------------------
    if not readable or not lang_ok or (ocr_confidence is not None and ocr_confidence < 0.6) or garbage_ratio(text) > 0.15:
        category = "document_processing_failure"
    elif not enough or (len(usable) == 0 and len(stated) == 0 and len(labs) == 0):
        category = "insufficient_data"
    elif not units_ok or no_unit or bad_labs or invalid or any(l.get("assumed_unit") for l in labs):
        category = "partial_extraction"
    else:
        category = "successful_extraction"

    # Boolean sub-checks ------------------------------------------------------------------------
    sub_checks = {
        "document_read": bool(readable and text.strip()),
        "language_identified": bool(lang_ok),
        "biomarkers_recognized": bool(len(labs) > 0 or len(stated) > 0),
        "values_extracted": bool(len(labs) > 0 and not bad_labs and not invalid),
        "units_identified": bool(units_ok and not any(l.get("assumed_unit") for l in labs)),
        "measurements_associated": bool(all(l.get("target") or l.get("analyte") for l in labs)),
        "temporal_distinction": bool(all(l.get("temporal") in ("current", "past", "indeterminate") for l in labs)),
        "conflicts_identified": bool(not any(l.get("ref_conflict") for l in labs) and not any(n.get("status") == "conflicting" for n in nutrients)),
    }

    # Summary counters --------------------------------------------------------------------------
    recognized_targets = set()
    for l in labs:
        if l.get("target"):
            recognized_targets.add(l["target"])
    for m in stated:
        if m.get("target"):
            recognized_targets.add(m["target"])

    num_deficient = sum(1 for n in nutrients if n.get("status") == "deficient")
    num_borderline = sum(1 for n in nutrients if n.get("status") == "borderline")
    num_normal = sum(1 for n in nutrients if n.get("status") in ("normal", "not_deficient"))
    num_missing = sum(1 for n in nutrients if n.get("status") == "not_assessed")
    num_review = sum(1 for n in nutrients if n.get("needs_review") or n.get("status") == "conflicting" or n.get("reliability") in ("verify", "abstain"))

    summary_counters = {
        "total_biomarkers_recognized": len(recognized_targets),
        "measurements_successfully_extracted": len(usable),
        "deficient": num_deficient,
        "borderline": num_borderline,
        "normal": num_normal,
        "missing": num_missing,
        "requiring_review": num_review,
    }

    pages = page_report or []
    out = {
        "category": category,
        "sub_checks": sub_checks,
        "summary_counters": summary_counters,
        "verdict": verdict,
        "message": {
            "ok": "The document passed all quality checks.",
            "verify": "Some values need to be verified against the original before relying on the result.",
            "insufficient_information": "There is not enough relevant information to assess any nutrient. This is a safe outcome, not an error.",
            "cannot_determine": "The system cannot reliably determine a result from this document. This is a safe outcome, not an error.",
        }[verdict],
        "checks": checks,
        "actions": list(dict.fromkeys(actions)),
        "abstained_nutrients": abstained,
        "pages_needing_attention": [p["page"] for p in pages if p.get("status") != "ok"],
    }
    return out
