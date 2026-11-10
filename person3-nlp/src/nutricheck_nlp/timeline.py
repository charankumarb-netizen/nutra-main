"""
Health Timeline Intelligence.

Reports from different months or years are analysed one by one; matching measurements are put on one timeline and compared
ONLY when it is safe to do so: the value is usable, has a known date, has a known (not assumed) unit that can be converted to
the same canonical unit, and is not a '<' / '>' value.  A downward trend is flagged FOR REVIEW; a trend is not a diagnosis.
Every point keeps its quote and location in the source report (evidence-backed).
"""

from __future__ import annotations

from typing import Any

from .lexicon import ANALYTES, display_unit

TREND_NOTE = "A trend is not a diagnosis. Values from different laboratories or methods may not be directly comparable."
REL_STEP, REL_TOTAL = 0.05, 0.10


def _key(iso: str) -> tuple[int, int, int]:
    p = [int(x) for x in iso.split("-")]
    return (p[0], p[1] if len(p) > 1 else 0, p[2] if len(p) > 2 else 0)


def collect_points(results: list[dict]) -> tuple[dict[str, list[dict]], list[dict]]:
    """-> (points by analyte, excluded entries with the reason).  `results` are analyze_text results with 'report_id'."""
    by: dict[str, list[dict]] = {}
    excluded: list[dict] = []
    for ri, res in enumerate(results):
        rid = res.get("report_id") or f"report-{ri + 1}"
        doc_date = (res.get("dates") or {}).get("document_date")
        for lab in res["labs"]:
            base = {"report_id": rid, "analyte": lab["analyte"], "label": lab["label"], "raw": lab["raw_text"],
                    "quote": lab["quote"], "location": lab["location"]}
            reason = None
            d = lab.get("date")
            if not lab["usable"]:
                reason = "value could not be verified (" + ", ".join(lab["verify_reasons"] or ["not usable"]) + ")"
            elif lab["assumed_unit"]:
                reason = "unit is missing, so the value is not comparable"
            elif lab["comparator"] not in (None, "="):
                reason = "value is given as '<' or '>' and cannot be compared numerically"
            elif d is None and doc_date and doc_date.get("iso"):
                d = {**doc_date, "source": "document"}
            if reason is None and (d is None or d.get("iso") is None):
                reason = "no usable date" if d is None else f"date '{d['raw']}' is ambiguous (day/month order unknown)"
            if reason:
                excluded.append({**base, "reason": reason})
                continue
            by.setdefault(lab["analyte"], []).append({
                **base, "date": d["iso"], "date_raw": d["raw"], "date_precision": d["precision"], "date_source": d.get("source"),
                "value": lab["value"], "unit": lab["unit"], "canonical_value": lab["canonical_value"],
                "canonical_unit": lab["canonical_unit"], "status": lab["status"],
                "converted": lab["unit"] != lab["canonical_unit"],
            })
    return by, excluded


def analyse_series(analyte: str, pts: list[dict]) -> dict[str, Any]:
    a = ANALYTES[analyte]
    pts = sorted(pts, key=lambda p: (_key(p["date"]), p["report_id"]))
    caveats: list[str] = []
    dates = [p["date"] for p in pts]
    same_day = {d for d in dates if dates.count(d) > 1}
    if same_day:
        caveats.append("More than one value on the same date (" + ", ".join(sorted(same_day)) + "): they are listed but not used to compute the trend.")
        seen: set[str] = set()
        trend_pts = []
        for p in pts:
            if p["date"] in same_day:
                continue
            trend_pts.append(p)
    else:
        trend_pts = list(pts)
    if any(p["date_precision"] != "day" for p in pts):
        caveats.append("Some dates are only month or year precise; the order is reliable but the interval is approximate.")
    if any(p["converted"] for p in pts):
        caveats.append("Some values were converted to " + display_unit(a.canonical_unit) + " using standard factors; confirm the original units.")
    if len({p["report_id"] for p in pts}) > 1:
        caveats.append("Different reports may come from different laboratories or methods.")
    out: dict[str, Any] = {
        "analyte": analyte, "label": a.label, "unit": display_unit(a.canonical_unit), "points": [
            {k: p[k] for k in ("date", "date_raw", "date_precision", "report_id", "value", "unit", "canonical_value", "status", "quote", "location", "converted")}
            for p in pts],
        "trend": "insufficient_data", "flag": False, "flag_text": "", "caveats": caveats, "note": TREND_NOTE,
    }
    if len(trend_pts) < 2:
        out["flag_text"] = "At least two dated, comparable measurements are needed to describe a trend."
        return out
    v = [p["canonical_value"] for p in trend_pts]
    first, last = v[0], v[-1]
    total = (last - first) / first if first else 0.0
    steps = [(v[i + 1] - v[i]) / v[i] if v[i] else 0.0 for i in range(len(v) - 1)]
    if all(s <= REL_STEP for s in steps) and total <= -REL_TOTAL:
        trend = "decreasing"
    elif all(s >= -REL_STEP for s in steps) and total >= REL_TOTAL:
        trend = "increasing"
    elif abs(total) < REL_TOTAL and all(abs(s) < 2 * REL_TOTAL for s in steps):
        trend = "stable"
    else:
        trend = "fluctuating"
    out.update(trend=trend, first={"date": trend_pts[0]["date"], "value": first}, last={"date": trend_pts[-1]["date"], "value": last},
               change_abs=round(last - first, 4), change_pct=round(total * 100, 1))
    low = [p for p in trend_pts if p["status"] in ("deficient", "borderline")]
    crossed = next((p for p in trend_pts if p["status"] in ("deficient", "borderline")), None)
    if trend == "decreasing":
        out["flag"] = True
        out["flag_text"] = (f"Downward trend for review: {a.label} went from {first:g} to {last:g} {display_unit(a.canonical_unit)} "
                            f"({out['change_pct']:+.0f}%) between {trend_pts[0]['date']} and {trend_pts[-1]['date']}.")
        if crossed and crossed is not trend_pts[0]:
            out["flag_text"] += f" The first value in the low/borderline range is from {crossed['date']}."
    elif trend == "fluctuating":
        out["flag"] = bool(low)
        out["flag_text"] = "The values go up and down; this is worth clarifying with the laboratory or a clinician."
    elif trend == "increasing":
        out["flag_text"] = f"Upward trend: {first:g} to {last:g} {display_unit(a.canonical_unit)}."
    else:
        out["flag_text"] = "Values are broadly stable."
    return out


def build_timeline(results: list[dict]) -> dict[str, Any]:
    by, excluded = collect_points(results)
    series = [analyse_series(a, p) for a, p in by.items()]
    series.sort(key=lambda s: (not s["flag"], s["label"]))
    undated = [e for e in excluded if "date" in e["reason"]]
    warnings = []
    if undated:
        warnings.append(f"{len(undated)} value(s) were left out of the timeline because the date is missing or ambiguous.")
    return {"series": series, "excluded": excluded, "warnings": warnings, "reports": len(results),
            "flagged": [s["analyte"] for s in series if s["flag"]], "note": TREND_NOTE}
