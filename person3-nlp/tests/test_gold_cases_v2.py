"""Second gold-case batch: negative, positive, borderline, context and advanced (193 cases)."""
import pytest

from gold_cases_v2 import ADVANCED_V2, BORDERLINE_V2, CONTEXT_V2, NEGATIVE_V2, POSITIVE_V2


def _run(engine, case):
    cid, text, expected = case[0], case[1], case[2]
    kwargs = case[3] if len(case) > 3 else {}
    res = engine.analyze_text(text, **kwargs)
    got = {n["key"]: n["status"] for n in res["nutrients"] + res["findings"]}
    wrong = {k: (v, got[k]) for k, v in expected.items() if got[k] != v}
    assert not wrong, f"[{cid}] {text!r}: (expected, got) = {wrong}"


@pytest.mark.parametrize("case", NEGATIVE_V2, ids=[c[0] for c in NEGATIVE_V2])
def test_negative_v2(engine, case):
    _run(engine, case)


@pytest.mark.parametrize("case", POSITIVE_V2, ids=[c[0] for c in POSITIVE_V2])
def test_positive_v2(engine, case):
    _run(engine, case)


@pytest.mark.parametrize("case", BORDERLINE_V2, ids=[c[0] for c in BORDERLINE_V2])
def test_borderline_v2(engine, case):
    _run(engine, case)


@pytest.mark.parametrize("case", CONTEXT_V2, ids=[c[0] for c in CONTEXT_V2])
def test_context_v2(engine, case):
    _run(engine, case)


@pytest.mark.parametrize("case", ADVANCED_V2, ids=[c[0] for c in ADVANCED_V2])
def test_advanced_v2(engine, case):
    _run(engine, case)


def test_v2_case_ids_are_unique_and_not_in_v1():
    from gold_cases import ALL_CASES
    ids = [c[0] for c in NEGATIVE_V2 + POSITIVE_V2 + BORDERLINE_V2 + CONTEXT_V2 + ADVANCED_V2]
    assert len(ids) == len(set(ids)) == 193
    assert not set(ids) & {c[0] for c in ALL_CASES}


def test_negative_cases_never_claim_deficiency(engine):
    """Safety property: a 'not deficient' expectation must never come back as deficient/borderline/suspected."""
    bad = []
    for cid, text, exp, *kw in NEGATIVE_V2:
        res = engine.analyze_text(text, **(kw[0] if kw else {}))
        got = {n["key"]: n["status"] for n in res["nutrients"] + res["findings"]}
        bad += [(cid, k, got[k]) for k, v in exp.items() if v in ("not_deficient", "not_assessed") and got[k] in ("deficient", "borderline", "suspected")]
    assert not bad, bad
