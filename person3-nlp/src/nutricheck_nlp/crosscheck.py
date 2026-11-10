"""
Contradiction and Missing-Information Detector.

Looks across ALL uploaded records of a person for information that is inconsistent, unclear or missing.
It never decides which record is right: it flags the difference and says what to ask to have it clarified.
"""

from __future__ import annotations

from typing import Any

from .lexicon import ANALYTES, TARGETS
from .quality import reason_text

SEV_ORDER = {"high": 0, "medium": 1, "info": 2}


def _ev(rid: str, quote: str, location: dict) -> dict:
    return {"report_id": rid, "quote": quote, "location": location}


def _doc_date(res: dict) -> str | None:
    d = (res.get("dates") or {}).get("document_date")
    return d.get("iso") if d else None


def cross_check(results: list[dict]) -> dict[str, Any]:
    issues: list[dict] = []

    def add(type_: str, severity: str, title: str, detail: str, evidence: list[dict], ask: str, target: str | None = None) -> None:
        issues.append({"id": f"I{len(issues) + 1}", "type": type_, "severity": severity, "target": target, "title": title,
                       "detail": detail, "evidence": evidence, "clarification_needed": ask})

    rids = [r.get("report_id") or f"report-{i + 1}" for i, r in enumerate(results)]

    # ---- 1. statements about a deficiency that disagree between reports --------------------------------------
    for key, tgt in TARGETS.items():
        stm: dict[str, list[tuple[str, dict, str | None]]] = {"affirmed": [], "denied": []}
        for rid, res in zip(rids, results):
            for m in res["mentions"]:
                if m["target"] == key and m["stance"] in stm:
                    stm[m["stance"]].append((rid, m, _doc_date(res)))
        for ra, ma, da in stm["affirmed"]:
            for rb, mb, db in stm["denied"]:
                if ra == rb:
                    continue
                differs = bool(da and db and da != db)
                add("statement_conflict", "medium" if differs else "high",
                    f"{tgt.name}: one report states a deficiency, another denies it",
                    ("The reports are dated differently (" + f"{da} vs {db}" + "), so this may be a real change over time, but the wording conflicts."
                     if differs else "The reports disagree and no dates explain the difference. The system did not decide which one is correct."),
                    [_ev(ra, ma["sentence"], ma["location"]), _ev(rb, mb["sentence"], mb["location"])],
                    f"Ask the clinician or laboratory to clarify whether {tgt.name.lower()} was low at the time of each report.", key)

    # ---- 2. symptom contradictions ---------------------------------------------------------------------------
    labels = {s["label"] for res in results for s in res.get("symptom_statements", [])}
    for lab in sorted(labels):
        pos, neg = [], []
        for rid, res in zip(rids, results):
            for s in res.get("symptom_statements", []):
                if s["label"] == lab:
                    (neg if s["assertion"] == "negated" else pos).append((rid, s, res))
        if pos and neg:
            ev = []
            for rid, s, res in pos[:1] + neg[:1]:
                ev.append(_ev(rid, s.get("sentence", s["text"]), s.get("location", {"offset": s["start"]})))
            add("symptom_conflict", "medium", f"Symptom “{lab}” is reported in one record and denied in another",
                "Symptoms can change over time, but the records give opposite answers.", ev,
                f"Ask whether “{lab}” was present at the time of each report.")

    # ---- 3. same test, same date, different values / unit differences ------------------------------------------
    seen: dict[tuple[str, str], list[tuple[str, dict]]] = {}
    for rid, res in zip(rids, results):
        for l in res["labs"]:
            if not l["usable"] or l["assumed_unit"] or l["comparator"] not in (None, "="):
                continue
            d = (l.get("date") or {}).get("iso") or _doc_date(res)
            if d:
                seen.setdefault((l["analyte"], d), []).append((rid, l))
    for (an, d), items in seen.items():
        vals = {round(l["canonical_value"], 2) for _, l in items}
        lo, hi = min(vals), max(vals)
        if len(vals) > 1 and lo and (hi - lo) / lo > 0.05:
            add("value_conflict", "high", f"{ANALYTES[an].label}: different values on the same date ({d})",
                "Two measurements dated the same day disagree by more than 5%.",
                [_ev(rid, l["quote"], l["location"]) for rid, l in items[:3]],
                "Ask which result is correct (a re-test, a transcription error or a different sample?).", ANALYTES[an].target)

    # ---- 4. lab value vs statement (within or across reports) ---------------------------------------------------
    for key in TARGETS:
        low_labs = [(rid, l) for rid, res in zip(rids, results) for l in res["labs"]
                    if l["target"] == key and l["usable"] and not l["assumed_unit"] and l["status"] == "deficient"]
        denials = [(rid, m) for rid, res in zip(rids, results) for m in res["mentions"] if m["target"] == key and m["stance"] == "denied" and m["polarity"] == "low"]
        for rl, l in low_labs[:1]:
            for rd, m in denials[:1]:
                add("lab_vs_statement", "high", f"{TARGETS[key].name}: a low laboratory value and a statement of no deficiency",
                    "The text says there is no deficiency but a laboratory value in the records is below the cut-off. Neither was silently preferred.",
                    [_ev(rl, l["quote"], l["location"]), _ev(rd, m["sentence"], m["location"])],
                    "Ask the clinician whether the statement refers to a different time, or the value is wrong.", key)

    # ---- 5. missing / unclear information --------------------------------------------------------------------------
    for rid, res in zip(rids, results):
        for l in res["labs"]:
            if l["verify_reasons"]:
                code = l["verify_reasons"][0]
                add("missing_or_unclear", "high" if code in ("unit_missing", "ambiguous_ocr_characters", "implausible_value") else "medium",
                    f"{l['label']} (“{l['raw_text']}”) {reason_text(code)}", "; ".join(l["warnings"]) or reason_text(code),
                    [_ev(rid, l["quote"], l["location"])], "Check this value against the original report or ask the laboratory.", l["target"])
        for r in res["rejected_values"]:
            if r["reason"] != "date_or_reference_range":
                add("missing_or_unclear", "high", f"Invalid value “{r['raw_text']}”", reason_text(r["reason"]),
                    [_ev(rid, r["raw_text"], r["location"])], "Check this value against the original report.", r.get("target"))
        if res["labs"] and not any((l.get("date") or {}).get("iso") for l in res["labs"]) and not _doc_date(res):
            add("missing_or_unclear", "info", "No date found in the report", "Without a date the results cannot be placed on a timeline.",
                [_ev(rid, "(whole report)", {})], "Add or confirm the collection date.")
        if any(l["analyte"] == "hemoglobin" for l in res["labs"]) and res["patient"]["sex"] is None:
            add("missing_or_unclear", "info", "Sex not stated (hemoglobin)", "The hemoglobin cut-off differs between men and women.",
                [_ev(rid, "(whole report)", {})], "Confirm the patient's sex so the right cut-off is used.", "anemia")
    # unit differences for the same analyte across reports
    units: dict[str, set[str]] = {}
    for res in results:
        for l in res["labs"]:
            if l["usable"] and not l["assumed_unit"]:
                units.setdefault(l["analyte"], set()).add(l["unit"])
    for an, u in units.items():
        if len(u) > 1:
            add("missing_or_unclear", "info", f"{ANALYTES[an].label} is reported in different units ({', '.join(sorted(u))})",
                "Values were converted with standard factors; check that the units were read correctly.", [], "Confirm the original units.", ANALYTES[an].target)

    # ---- 6. contradictions found INSIDE one report ---------------------------------------------------------------
    for rid, res in zip(rids, results):
        for n in res["nutrients"] + res["findings"]:
            for reason in n["review_reasons"]:
                if reason in ("text_contradicts_lab", "conflicting_measurements"):
                    ev = [_ev(rid, e["quote"], e["location"]) for e in n["detective"]["evidence"][:3]]
                    add("within_report_conflict", "high", f"{n['name']}: " + ("text and laboratory value disagree" if reason == "text_contradicts_lab"
                                                                         else "several different values without dates"),
                        "; ".join(n["notes"]) or "The report contradicts itself.", ev,
                        "Ask for clarification; the system did not choose a side.", n["key"])
    issues.sort(key=lambda i: (SEV_ORDER[i["severity"]], i["id"]))
    return {"issues": issues, "count": len(issues), "by_severity": {s: sum(1 for i in issues if i["severity"] == s) for s in SEV_ORDER},
            "principle": "The system flags differences and missing information; it does not decide which record is correct."}


def _sentence_at(res: dict, offset: int) -> str:
    for s in res.get("symptom_statements", []):
        if s["start"] == offset:
            return s["text"]
    return ""
