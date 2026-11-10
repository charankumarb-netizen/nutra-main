"""
Explain My Report at My Level.

Plain-language explanations at five levels (very simple English, detailed English, medical terminology with definitions,
Telugu, Hindi).  Every sentence is generated from a fixed template + the structured findings of the English engine, and every
sentence carries evidence references (E1, E2, ...) that point at the exact quote / value / page in the ORIGINAL report.

Why templates and not free generation: a translation or simplification must never silently change medical meaning.
Numbers, units, dates and quotes are inserted programmatically (never translated); `verify_explanation` re-checks every
number, quote and status against the analysis result and flags any sentence that does not match.
"""

from __future__ import annotations

import re
from typing import Any

from . import i18n
from .lexicon import ANALYTES, HEMOGLOBIN_THRESHOLD_BY_SEX, TARGETS, display_unit

_NUM = re.compile(r"\d+(?:\.\d+)?")
_QUOTE = re.compile(r"“([^”]*)”")


def _g(x: float | None) -> str:
    return "" if x is None else f"{round(x, 2):g}"


def _nums(*parts: Any) -> set[str]:
    out: set[str] = set()
    for p in parts:
        if p is None:
            continue
        out.update(_NUM.findall(str(p)))
        out.update(str(float(n)).rstrip("0").rstrip(".") if "." in n else n for n in _NUM.findall(str(p)))
        out.update(f"{float(n):g}" for n in _NUM.findall(str(p)))
        out.update(f"{round(float(n), 2):g}" for n in _NUM.findall(str(p)))
    return out


def _names(level: str, key: str) -> str:
    if level == "hi":
        return i18n.NAME_HI[key]
    if level == "te":
        return i18n.NAME_TE[key]
    return i18n.NAME_EN[key]


def _function(level: str, key: str) -> str:
    if level == "hi":
        return i18n.FUNCTION_HI[key]
    if level == "te":
        return i18n.FUNCTION_TE[key]
    if level in ("detailed_en", "medical_en"):
        return i18n.FUNCTION_DETAILED[key]
    return i18n.FUNCTION_SIMPLE[key]


def _cutoffs(lab: dict, sex: str | None) -> tuple[float, float | None]:
    a = ANALYTES[lab["analyte"]]
    low = a.deficient_below
    if lab["analyte"] == "hemoglobin":
        low = HEMOGLOBIN_THRESHOLD_BY_SEX.get(sex, 12.0)
    return low, a.borderline_below


def build_evidence_index(result: dict) -> tuple[list[dict], dict[tuple, str]]:
    """Number the pieces of evidence of the whole analysis: E1, E2, ... (stable order)."""
    index: list[dict] = []
    lookup: dict[tuple, str] = {}
    for n in result["nutrients"] + result["findings"]:
        for ev in n["detective"]["evidence"]:
            if ev["kind"] not in ("lab", "statement", "lab_not_used"):
                continue
            k = (n["key"], ev["kind"], ev["span"]["start"], ev["span"]["end"])
            if k in lookup:
                continue
            eid = f"E{len(index) + 1}"
            lookup[k] = eid
            index.append({"id": eid, "target": n["key"], "kind": ev["kind"], "found": ev["found"], "quote": ev["quote"],
                          "location_text": ev["location_text"], "date": ev.get("date"), "cutoff": ev.get("cutoff"),
                          "span": ev["span"]})
    return index, lookup


def explain(result: dict, level: str = "simple_en") -> dict[str, Any]:
    if level not in i18n.LEVELS:
        raise ValueError(f"level must be one of {sorted(i18n.LEVELS)}")
    S = i18n.SENTENCES["simple_en" if level == "simple_en" else level]
    evidence, lookup = build_evidence_index(result)
    sex = result.get("patient", {}).get("sex")
    sections: list[dict] = []
    glossary_terms: list[str] = []

    for n in result["nutrients"] + result["findings"]:
        key, status = n["key"], n["status"]
        if status == "not_assessed" and not n["review_reasons"]:
            continue
        name = _names(level, key)
        paras: list[dict] = []

        def add(text: str, refs: list[str], numbers: set[str] | None = None, kind: str = "statement") -> None:
            paras.append({"text": text, "kind": kind, "status": status, "evidence_refs": refs})

        add(_function(level, key), [], kind="background")
        labs = [e for e in n["detective"]["evidence"] if e["kind"] == "lab"]
        stmts = [e for e in n["detective"]["evidence"] if e["kind"] == "statement"]
        old = [e for e in n["detective"]["evidence"] if e["kind"] == "lab_not_used"]
        used_labs = {(l["start"], l["end"]): l for l in result["labs"]}
        first_quote_done = False
        for e in labs:
            lab = used_labs.get((e["span"]["start"], e["span"]["end"]))
            if lab is None or not lab["usable"]:
                continue
            ref = lookup[(key, e["kind"], e["span"]["start"], e["span"]["end"])]
            low, border = _cutoffs(lab, sex)
            k = {"deficient": "lab_deficient", "borderline": "lab_borderline", "normal": "lab_normal", "high": "lab_high"}.get(lab["status"])
            if k is None:
                continue
            if lab["provisional"]:
                add({"hi": "इस मान की इकाई रिपोर्ट में नहीं है, इसलिए इसे केवल अस्थायी माना गया है।",
                     "te": "ఈ విలువకు యూనిట్ రిపోర్ట్‌లో లేదు, కాబట్టి దీన్ని తాత్కాలికంగానే పరిగణించాము."}.get(
                         level, "The report gives no unit for this value, so it is treated as provisional only."), [ref], kind="caveat")
            slots = dict(name=name, test=lab["label"], value=_g(lab["canonical_value"]), unit=display_unit(lab["canonical_unit"]),
                         cutoff=_g(low), cutoff_lo=_g(low), cutoff_hi=_g(border if border is not None else low))
            if lab["status"] == "borderline":
                slots["cutoff_lo"], slots["cutoff_hi"] = _g(low), _g(border)
            add(S[k].format(**slots), [ref], kind="lab")
            if lab["unit"] != lab["canonical_unit"]:
                add({"hi": "रिपोर्ट में यह {v} {u} के रूप में लिखा है।", "te": "రిపోర్ట్‌లో ఇది {v} {u} గా ఉంది."}.get(
                    level, "(The report states it as {v} {u}.)").format(v=_g(lab["value"]), u=display_unit(lab["unit"])), [ref], kind="lab")
            if e.get("date"):
                add(S["date"].format(date=e["date"]), [ref], kind="date")
            if not first_quote_done or level in ("detailed_en", "medical_en"):
                add(S["quote"].format(quote=e["quote"]), [ref], kind="quote")
                first_quote_done = True
            if lab["verify_reasons"]:
                add(S["verify"], [ref], kind="caveat")
        for e in stmts:
            ref = lookup[(key, e["kind"], e["span"]["start"], e["span"]["end"])]
            if not first_quote_done:
                add(S["quote"].format(quote=e["quote"]), [ref], kind="quote")
                first_quote_done = True
        refs_all = [lookup[(key, e["kind"], e["span"]["start"], e["span"]["end"])] for e in labs + stmts if (key, e["kind"], e["span"]["start"], e["span"]["end"]) in lookup]
        if status in ("deficient", "suspected", "not_deficient", "conflicting", "not_assessed") and not (labs and status in ("deficient", "not_deficient") and level not in ("medical_en",)):
            add(S[status].format(name=name), refs_all, kind="status")
        elif labs == [] and status == "borderline":
            add(S["suspected"].format(name=name), refs_all, kind="status")
        if n["history_of_deficiency"]:
            add(S["history"], [lookup[(key, e["kind"], e["span"]["start"], e["span"]["end"])] for e in old if (key, e["kind"], e["span"]["start"], e["span"]["end"]) in lookup], kind="history")
        if n["needs_review"] and not any(p["kind"] == "caveat" for p in paras):
            add(S["verify"], refs_all, kind="caveat")
        sections.append({"key": key, "name": name, "status": status, "reliability": n["reliability"], "paragraphs": paras})
        if level == "medical_en":
            glossary_terms += i18n.TERMS_BY_STATUS_AND_KEY.get(key, [])

    if not sections:
        sections.append({"key": None, "name": "", "status": "not_assessed", "reliability": "abstain", "paragraphs": [
            {"text": {"hi": "इस रिपोर्ट में पोषक तत्वों के बारे में कुछ कहने के लिए पर्याप्त जानकारी नहीं मिली।",
                      "te": "ఈ రిపోర్ట్‌లో పోషకాల గురించి చెప్పడానికి తగినంత సమాచారం కనబడలేదు."}.get(
                          level, "This report does not contain enough information about the target nutrients to explain."),
             "kind": "status", "status": "not_assessed", "evidence_refs": []}]})

    q_lang = "hi" if level == "hi" else "te" if level == "te" else "en"
    questions = [i18n.QUESTIONS[k][q_lang] for k in ("q_range", "q_compare", "q_context", "q_repeat", "q_medicines")]
    out: dict[str, Any] = {
        "level": level, **i18n.LEVELS[level], "title": S["title"], "sections": sections, "evidence": evidence,
        "questions_title": S["questions_title"], "questions": questions, "disclaimer": S["disclaimer"],
        "glossary": [{"term": t, "definition": i18n.GLOSSARY[t]} for t in dict.fromkeys(glossary_terms)] if level == "medical_en" else [],
        "notice": i18n.TRANSLATION_NOTICE.get(level),
    }
    out["traceability"] = verify_explanation(out, result)
    return out


def verify_explanation(expl: dict, result: dict) -> dict[str, Any]:
    """Re-check an explanation against the analysis: evidence, numbers, quotes and statuses must all match. -> {"ok", "problems"}"""
    problems: list[str] = []
    index, _ = build_evidence_index(result)
    truth = {e["id"]: e for e in index}
    for e in expl.get("evidence", []):
        t = truth.get(e["id"])
        if t is None or any(t[k] != e.get(k) for k in ("quote", "found", "location_text", "date")):
            problems.append(f"{e.get('id')}: evidence does not match the analysis")
    status_by_key = {n["key"]: n["status"] for n in result["nutrients"] + result["findings"]}
    names_numbers = _nums(*i18n.NAME_HI.values(), *i18n.NAME_TE.values(), *[a.label for a in ANALYTES.values()],
                          *[t.name for t in TARGETS.values()])
    for sec in expl.get("sections", []):
        if sec.get("key") is not None and sec.get("status") != status_by_key.get(sec["key"]):
            problems.append(f"{sec['key']}: section status {sec.get('status')!r} != analysis status {status_by_key.get(sec['key'])!r}")
        for p in sec["paragraphs"]:
            if p.get("kind") != "background" and p.get("status") != sec.get("status"):
                problems.append(f"{sec.get('key')}: paragraph status mismatch")
            refs = [truth.get(r) for r in p["evidence_refs"]]
            if any(r is None for r in refs):
                problems.append(f"{sec.get('key')}: unknown evidence reference in “{p['text'][:40]}…”")
                continue
            allowed = set(names_numbers)
            for r in refs:
                allowed |= _nums(r["found"], r["quote"], r["date"], r["cutoff"])
            for r in refs:                                     # cut-offs of every analyte of this target are allowed
                for a in ANALYTES.values():
                    if a.target == r["target"]:
                        allowed |= _nums(a.deficient_below, a.borderline_below, a.high_above)
            allowed |= _nums(*HEMOGLOBIN_THRESHOLD_BY_SEX.values())
            visible = _QUOTE.sub(" ", p["text"])
            extra = {x for x in _NUM.findall(visible) if x not in allowed and f"{float(x):g}" not in allowed}
            if extra and p.get("kind") != "background":
                problems.append(f"{sec.get('key')}: number(s) {sorted(extra)} in “{p['text'][:50]}…” are not in the referenced evidence")
            for q in _QUOTE.findall(p["text"]):
                if not any(q in r["quote"] for r in refs if r):
                    problems.append(f"{sec.get('key')}: quoted text is not in the referenced evidence")
    return {"ok": not problems, "problems": problems,
            "statement": "Every number, quote and status in this explanation was re-checked against the analysis of the original report."}
