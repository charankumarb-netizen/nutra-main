"""
Medical Report Detective: turns a finding into an inspectable chain of evidence.

For every nutrient it answers four questions:
    * What evidence was found?        (exact value + unit, the sentence it came from, its date)
    * Where is it?                    (page, section heading, line, character span for highlighting)
    * What is the context?            (supplements mentioned or denied, symptoms, risk factors, history)
    * What information is missing?    (reference range, date, unit, related tests, clinical context)
"""

from __future__ import annotations

import re
from typing import Any

from .lexicon import ANALYTES, TARGETS, display_unit

_HEADING_WORDS = re.compile(
    r"\b(?:laborator\w*|lab|results?|findings?|assessment|plan|impression|diagnos[ie]s|history|medications?|medicines?|"
    r"examination|exam|summary|complaints?|hpi|investigations?|tests?|blood|h(?:a)?ematology|biochemistry|chemistry|vitamins?|"
    r"nutrition|follow[\s-]?up|clinical|notes?|recommendations?|comments?|review|systems|presenting|illness)\b", re.I)
_HEADING_LINE = re.compile(r"^(?:#+\s*)?(?P<h>[A-Za-z][A-Za-z /&,()'-]{2,58}?)\s*:?\s*$")
MAX_QUOTE = 240


def line_bounds(text: str, offset: int) -> tuple[int, int]:
    s = text.rfind("\n", 0, offset) + 1
    e = text.find("\n", offset)
    return s, (len(text) if e == -1 else e)


def section_for(text: str, offset: int) -> str | None:
    """Nearest heading ABOVE `offset` (e.g. 'Laboratory results'), or None."""
    pos = text.rfind("\n", 0, offset)
    steps = 0
    while pos > 0 and steps < 80:
        prev = text.rfind("\n", 0, pos)
        line = text[prev + 1:pos].strip()
        if line:
            m = _HEADING_LINE.match(line)
            if m:
                h = m.group("h").strip()
                words = h.split()
                if len(words) <= 6 and (h.isupper() or line.endswith(":") or line.startswith("#") or _HEADING_WORDS.search(h)):
                    return h.lower().strip(" :")
        pos, steps = prev, steps + 1
        if prev < 0:
            break
    return None


def quote_for(text: str, start: int, end: int) -> str:
    s, e = line_bounds(text, start)
    q = text[s:e].strip()
    if len(q) > MAX_QUOTE:
        a = max(0, start - s - 80)
        q = "…" + text[s + a:s + a + MAX_QUOTE].strip() + "…"
    return q


def location_text(loc: dict, section: str | None) -> str:
    parts = []
    parts.append(f"page {loc['page']}" if loc.get("page") is not None else "page not applicable")
    if section:
        parts.append(f"section “{section}”")
    parts.append(f"line {loc['line']}")
    return ", ".join(parts)


def fmt_date(d: dict | None) -> str | None:
    if not d:
        return None
    if d.get("ambiguous"):
        return f"{d['raw']} (ambiguous day/month order)"
    return d.get("iso") or d.get("raw")


def lab_found_text(l: dict) -> str:
    cmp_ = l.get("comparator") or ""
    base = f"{l['label']} value of {cmp_}{l['value']:g} {display_unit(l['unit'])}"
    if l.get("unit") != l.get("canonical_unit") and l.get("canonical_value") is not None:
        base += f" (= {l['canonical_value']:g} {display_unit(l['canonical_unit'])})"
    return base


def build_detective(
    key: str,
    used_labs: list[dict],
    other_labs: list[dict],
    mentions: list[dict],
    treatments: list[dict],
    symptoms: list[dict],
    risks: list[dict],
    text: str,
    sex: str | None,
    history: bool,
    rejected: list[dict],
) -> dict[str, Any]:
    tgt = TARGETS[key]
    evidence: list[dict] = []
    for l in used_labs + other_labs:
        loc = l["location"]
        a = ANALYTES[l["analyte"]]
        status_txt = {"deficient": "low (below the cut-off)", "borderline": "borderline", "normal": "within range",
                      "high": "above range (not a deficiency)", "indeterminate": "could not be classified"}[l["status"]]
        evidence.append({
            "kind": "lab" if l in used_labs else "lab_not_used",
            "found": lab_found_text(l) + f" → {status_txt}",
            "quote": quote_for(text, l["start"], l["end"]),
            "location": loc,
            "location_text": location_text(loc, loc.get("section")),
            "date": fmt_date(l.get("date")),
            "temporal": l.get("temporal"),
            "span": {"start": l["start"], "end": l["end"]},
            "lab_reference": l.get("lab_reference"),
            "cutoff": f"deficient < {a.deficient_below:g} {display_unit(a.canonical_unit)}" + (f"; borderline < {a.borderline_below:g}" if a.borderline_below else ""),
            "needs_verification": bool(l.get("verify_reasons")),
            "verify_reasons": l.get("verify_reasons", []),
        })
    for m in mentions:
        loc = m["location"]
        evidence.append({
            "kind": "statement",
            "found": f"Statement “{m['text']}” (read as: {m['assertion']}; polarity: {m['polarity']})",
            "quote": m["sentence"][:MAX_QUOTE],
            "location": loc,
            "location_text": location_text(loc, loc.get("section")),
            "date": None, "temporal": None,
            "span": {"start": m["start"], "end": m["end"]},
            "needs_verification": False, "verify_reasons": [],
        })
    for r in rejected:
        if r.get("target") == key and r["reason"] not in ("date_or_reference_range",):
            evidence.append({
                "kind": "rejected_value",
                "found": f"Value “{r['raw_text']}” was rejected: {r['reason'].replace('_', ' ')}",
                "quote": quote_for(text, r["start"], r["end"]),
                "location": r.get("location", {}), "location_text": r.get("location_text", ""),
                "date": None, "temporal": None, "span": {"start": r["start"], "end": r["end"]},
                "needs_verification": True, "verify_reasons": [r["reason"]],
            })

    # ---- context -----------------------------------------------------------------------------
    context: list[str] = []
    mine_t = [t for t in treatments if t["target"] == key]
    taking = [t for t in mine_t if t["assertion"] != "negated"]
    denied = [t for t in mine_t if t["assertion"] == "negated"]
    for t in taking:
        context.append(f"Supplement / treatment mentioned: “{t['text']}”.")
    for t in denied:
        context.append(f"Supplement use is denied: “{t['text']}”.")
    if not mine_t and tgt.kind == "nutrient":
        context.append(f"No {tgt.name.lower()} supplementation or treatment is mentioned in the report.")
    for s in symptoms:
        if key in s["targets"]:
            context.append(f"Symptom {'denied' if s['assertion'] == 'negated' else 'mentioned'}: {s['label']}.")
    for r in risks:
        if key in r["targets"]:
            context.append(f"Risk factor mentioned: {r['label']}.")
    if history:
        context.append("An earlier (historical) low result or deficiency is mentioned; it is kept separate from the current result.")

    # ---- missing information -----------------------------------------------------------------
    missing: list[str] = []
    for l in used_labs:
        if l.get("lab_reference") is None:
            missing.append(f"The laboratory's own reference range for {l['label']} is not shown.")
        if not l.get("date"):
            missing.append(f"No collection or test date was found for {l['label']}.")
        if l.get("assumed_unit"):
            missing.append(f"The unit for {l['label']} is missing (it was assumed, so the result is provisional).")
    if not used_labs and not other_labs and mentions:
        missing.append("No laboratory value in the report supports or contradicts the written statement.")
    low = re.compile(r"(?<![a-z])albumin")
    lowered = text.lower()
    if key == "calcium" and (used_labs or other_labs) and not low.search(lowered):
        missing.append("No albumin value was found (needed to interpret total calcium).")
    if key == "iron" and used_labs and all(l["analyte"] == "ferritin" for l in used_labs) and "transferrin" not in lowered:
        missing.append("No transferrin saturation was found (ferritin alone can be affected by inflammation).")
    if key == "vitamin_b12" and any(l["status"] == "borderline" for l in used_labs) and not re.search(r"methylmalonic|homocysteine", lowered):
        missing.append("No methylmalonic acid or homocysteine result was found (often used when B12 is borderline).")
    if key == "anemia" and sex is None and used_labs:
        missing.append("The patient's sex is not stated (the hemoglobin cut-off differs by sex).")
    if tgt.kind == "nutrient" and (used_labs or mentions) and not (symptoms or risks or mine_t):
        missing.append("No symptoms, medicines, diet or medical history that could help interpret the result were found.")
    return {"evidence": evidence, "context": list(dict.fromkeys(context)), "missing_information": list(dict.fromkeys(missing))}
