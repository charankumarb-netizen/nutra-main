"""The laboratory's OWN reference interval: parsed, compared, flagged - never silently overriding the clinical cut-off."""
import pytest


def lab(res, analyte):
    return next(l for l in res["labs"] if l["analyte"] == analyte)


def nut(res, key):
    return next(n for n in res["nutrients"] + res["findings"] if n["key"] == key)


@pytest.mark.parametrize("text,low,high", [
    ("Ferritin: 25 ng/mL (Ref: 30 - 400)", 30, 400),
    ("Ferritin 8 ng/mL 15 - 150", 15, 150),
    ("Vitamin D: 18 ng/mL Ref: 20 to 100", 20, 100),
    ("Vitamin B12: 400 pg/mL (Reference: 200 - 900 pg/mL)", 200, 900),
    ("Calcium total: 9.1 mg/dL [Reference range: 8.5 - 10.5]", 8.5, 10.5),
    ("Ferritin 12 µg/L (Ref: 15 - 150 ng/mL)", 15, 150),
])
def test_reference_interval_parsed(engine, text, low, high):
    res = engine.analyze_text(text)
    ref = res["labs"][0]["lab_reference"]
    assert ref and ref["low"] == pytest.approx(low, rel=1e-3) and ref["high"] == pytest.approx(high, rel=1e-3)


def test_reference_keeps_original_text(engine):
    ref = lab(engine.analyze_text("Ferritin: 25 ng/mL (Ref: 30 - 400)"), "ferritin")["lab_reference"]
    assert "Ref" in ref["raw"] and "30 - 400" in ref["raw"]        # original case, not the lower-cased copy


def test_value_inside_below_above(engine):
    assert lab(engine.analyze_text("Ferritin 90 ng/mL (Ref: 30 - 400)"), "ferritin")["reference_check"] == "within"
    assert lab(engine.analyze_text("Ferritin 8 ng/mL (Ref: 30 - 400)"), "ferritin")["reference_check"] == "below"
    assert lab(engine.analyze_text("Ferritin 500 ng/mL (Ref: 30 - 400)"), "ferritin")["reference_check"] == "above"


def test_no_reference_means_no_check_and_no_flag(engine):
    res = engine.analyze_text("Ferritin: 90 ng/mL")
    assert lab(res, "ferritin")["lab_reference"] is None and lab(res, "ferritin")["reference_check"] is None
    assert not res["needs_human_review"]


def test_normal_by_cutoff_but_below_lab_interval_is_flagged_not_changed(engine):
    """Vitamin D 32: normal by the 30 ng/mL cut-off, but below THIS lab's own 40-100 target -> keep the status, ask for review."""
    res = engine.analyze_text("Vitamin D: 32 ng/mL (Ref: 40 - 100)")
    assert nut(res, "vitamin_d")["status"] == "not_deficient"
    assert res["needs_human_review"] and nut(res, "vitamin_d")["needs_review"]
    assert "lab_reference_disagreement" in nut(res, "vitamin_d")["review_reasons"]
    assert any("reference range" in w for w in res["warnings"])


def test_deficient_by_cutoff_but_inside_lab_interval_is_flagged(engine):
    res = engine.analyze_text("Vitamin D: 18 ng/mL (Ref: 10 - 100)")
    assert nut(res, "vitamin_d")["status"] == "deficient"
    assert "lab_reference_disagreement" in nut(res, "vitamin_d")["review_reasons"] and res["needs_human_review"]


def test_consistent_reference_causes_no_review(engine):
    res = engine.analyze_text("Ferritin: 8 ng/mL (Ref: 15 - 150)")
    assert nut(res, "iron")["status"] == "deficient" and not nut(res, "iron")["needs_review"]


@pytest.mark.parametrize("text", [
    "Hemoglobin: 13.7 g/dL [ref.: > 20]",                    # a lower limit above the analyte's own "high" threshold
    "Calcium: 9.9 mg/dL (Ref: 15 - 150)",                       # another analyte's interval
    "Vitamin B12: 600 pg/mL (Reference range: 30 - 100 ng/mL)",   # unit mix-up
    "Ferritin 90 ng/mL > 20",                                # a bare comparator without a marker is ambiguous
])
def test_implausible_or_ambiguous_references_are_ignored(engine, text):
    res = engine.analyze_text(text)
    assert res["labs"][0]["lab_reference"] is None
    assert not any(n["needs_review"] for n in res["nutrients"] + res["findings"])


def test_reference_range_is_never_read_as_a_value(engine):
    res = engine.analyze_text("Ferritin (Ref: 30 - 400 ng/mL)")
    assert res["labs"] == [] and nut(res, "iron")["status"] == "not_assessed"
