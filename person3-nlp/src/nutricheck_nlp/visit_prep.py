"""
Doctor-Visit Preparation Assistant.

Creates a one-page summary (Markdown + printable HTML) from the analysis: extracted results with dates and sources, changes over
time, items that need clarification, and questions to ask.  It prepares the person for an appointment; it does not diagnose and
never recommends a supplement or dose.
"""

from __future__ import annotations

import html
from typing import Any

from . import i18n
from .lexicon import ANALYTES, TARGETS, display_unit

STATUS_TEXT = {"deficient": "low", "borderline": "borderline", "normal": "within range", "high": "above range", "indeterminate": "not classified"}


def specific_questions(results: list[dict], timeline: dict | None, crosscheck: dict | None) -> list[str]:
    qs: list[str] = []
    for res in results:
        for n in res["nutrients"] + res["findings"]:
            nm = TARGETS[n["key"]].name
            if n["status"] == "deficient":
                qs.append(f"My {nm.lower()} result is below the cut-off. What could explain it, and which other tests (if any) would help find the cause?")
            elif n["status"] == "borderline":
                qs.append(f"My {nm.lower()} result is borderline. Should it be re-checked, and when?")
            elif n["status"] == "suspected":
                qs.append(f"The report only suggests a possible {nm.lower()} problem. What would confirm or exclude it?")
            elif n["status"] == "conflicting":
                qs.append(f"My records give conflicting information about {nm.lower()}. Which result should I rely on?")
            if n["history_of_deficiency"]:
                qs.append(f"An earlier low {nm.lower()} result is mentioned. Is the current result consistent with the earlier one?")
            for m in n["detective"]["missing_information"]:
                if "albumin" in m:
                    qs.append("Should my calcium be interpreted together with albumin or ionized calcium?")
                if "transferrin" in m:
                    qs.append("Would a transferrin saturation test add useful information to my ferritin?")
                if "methylmalonic" in m:
                    qs.append("Would methylmalonic acid or homocysteine testing help clarify my vitamin B12 result?")
                if "sex is not stated" in m:
                    qs.append("Is the hemoglobin reference range for my sex and age the one used here?")
    if timeline:
        for s in timeline["series"]:
            if s["flag"]:
                qs.append(f"My {s['label']} appears to have changed over time ({s['trend']}). Is that clinically meaningful?")
    if crosscheck:
        for i in crosscheck["issues"]:
            if i["severity"] == "high":
                qs.append(i["clarification_needed"])
    seen: set[str] = set()
    uniq = [q for q in qs if not (q in seen or seen.add(q))]
    return uniq[:10]


def build_visit_prep(results: list[dict], timeline: dict | None = None, crosscheck: dict | None = None,
                     language: str = "en", generated_on: str | None = None) -> dict[str, Any]:
    if language not in ("en", "hi", "te"):
        raise ValueError("language must be 'en', 'hi' or 'te'")
    rows = []
    for ri, res in enumerate(results):
        rid = res.get("report_id") or f"report-{ri + 1}"
        for l in res["labs"]:
            d = (l.get("date") or {}).get("iso") or ((res.get("dates") or {}).get("document_date") or {}).get("iso")
            rows.append({"report": rid, "test": l["label"], "value": f"{l['comparator'] or ''}{l['value']:g}", "unit": display_unit(l["unit"]),
                         "date": d or "not found",
                         "reading": STATUS_TEXT[l["status"]] + (" (verify)" if l["verify_reasons"] else ""),
                         "source": f"page {l['location']['page'] or 'n/a'}, line {l['location']['line']}"})
    changes = [s for s in (timeline or {"series": []})["series"] if len(s["points"]) >= 2]
    clar = [i for i in (crosscheck or {"issues": []})["issues"] if i["severity"] in ("high", "medium")]
    unclear = []
    for res in results:
        unclear += res["quality"]["actions"]
    unclear = list(dict.fromkeys(unclear))
    generic = [i18n.QUESTIONS[k][language] for k in ("q_range", "q_compare", "q_context", "q_repeat", "q_medicines")]
    specific = specific_questions(results, timeline, crosscheck)
    title = {"en": "Doctor-visit summary", "hi": "डॉक्टर से मुलाक़ात का सारांश", "te": "డాక్టర్ సందర్శన సారాంశం"}[language]

    md = [f"# {title}", ""]
    if generated_on:
        md += [f"_Prepared on {generated_on}_", ""]
    md += ["> This summary helps you prepare for an appointment. It is not a diagnosis and does not recommend any treatment or supplement.", ""]
    md += ["## Extracted results", ""]
    if rows:
        md += ["| Test | Result | Date | Reading | Source |", "|---|---|---|---|---|"]
        md += [f"| {r['test']} | {r['value']} {r['unit']} | {r['date']} | {r['reading']} | {r['report']}: {r['source']} |" for r in rows]
    else:
        md += ["No laboratory values were extracted."]
    md += ["", "## Changes over time", ""]
    if changes:
        for s in changes:
            pts = []
            for p in s.get("points", []):
                p_date = p.get("date", "")
                p_val = p.get("canonical_value", "")
                pts.append(f"{p_date}: {p_val}")
            pts_str = ", ".join(pts)
            flag_or_trend = s.get("flag_text") or s.get("trend", "")
            s_label = s.get("label", "")
            s_unit = s.get("unit", "")
            md.append(f"- **{s_label}**: {flag_or_trend} ({pts_str} {s_unit})")
    else:
        md.append("Only one dated report was available, so no change can be described.")
    md += ["", "## Items to clarify", ""]
    md += [f"- {c['title']} — {c['clarification_needed']}" for c in clar] + [f"- {u}" for u in unclear] or ["Nothing flagged."]
    md += ["", "## Questions to discuss", ""]
    md += [f"{n}. {q}" for n, q in enumerate(generic + ([] if language != "en" else specific), 1)]
    if language != "en" and specific:
        md += ["", "_Additional case-specific questions (English):_"] + [f"- {q}" for q in specific]
    markdown = "\n".join(md)

    e = html.escape
    hh = [f"<!doctype html><html lang='{language}'><head><meta charset='utf-8'><title>{e(title)}</title>",
          "<style>body{font:13px/1.4 system-ui,sans-serif;max-width:780px;margin:12mm auto;color:#111}h1{font-size:20px}h2{font-size:15px;margin:14px 0 4px;border-bottom:1px solid #ccc}"
          "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:3px 6px;text-align:left}.note{background:#f4f4f4;padding:6px}"
          "@media print{body{margin:8mm}}</style></head><body>", f"<h1>{e(title)}</h1>"]
    if generated_on:
        hh.append(f"<p><i>Prepared on {e(generated_on)}</i></p>")
    hh.append("<p class='note'>This summary helps you prepare for an appointment. It is not a diagnosis and does not recommend any treatment or supplement.</p>")
    hh.append("<h2>Extracted results</h2>")
    if rows:
        hh.append("<table><tr><th>Test</th><th>Result</th><th>Date</th><th>Reading</th><th>Source</th></tr>")
        for r in rows:
            hh.append(f"<tr><td>{e(r['test'])}</td><td>{e(r['value'])} {e(r['unit'])}</td><td>{e(r['date'])}</td><td>{e(r['reading'])}</td><td>{e(r['report'])}: {e(r['source'])}</td></tr>")
        hh.append("</table>")
    else:
        hh.append("<p>No laboratory values were extracted.</p>")
    hh.append("<h2>Changes over time</h2><ul>")
    hh += [f"<li><b>{e(s['label'])}</b>: {e(s['flag_text'] or s['trend'])}</li>" for s in changes] or ["<li>Only one dated report was available.</li>"]
    hh.append("</ul><h2>Items to clarify</h2><ul>")
    hh += [f"<li>{e(c['title'])} — {e(c['clarification_needed'])}</li>" for c in clar] + [f"<li>{e(u)}</li>" for u in unclear] or ["<li>Nothing flagged.</li>"]
    hh.append("</ul><h2>Questions to discuss</h2><ol>")
    hh += [f"<li>{e(q)}</li>" for q in generic + specific]
    hh.append("</ol></body></html>")
    return {"language": language, "title": title, "markdown": markdown, "html": "".join(hh), "questions": generic + specific,
            "sections": {"results": rows, "changes": [s["analyte"] for s in changes], "to_clarify": [c["title"] for c in clar] + unclear}}
