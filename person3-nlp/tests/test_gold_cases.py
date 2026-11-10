"""Hand-written gold cases: positive, borderline, negative, context and advanced."""
import pytest

from gold_cases import ADVANCED, BORDERLINE, CONTEXT, NEGATIVE, POSITIVE


def _check(statuses, case_id, text, expected):
    got, _ = statuses(text)
    wrong = {k: (v, got[k]) for k, v in expected.items() if got[k] != v}
    assert not wrong, f"[{case_id}] {text!r}: (expected, got) = {wrong}"


@pytest.mark.parametrize("cid,text,expected", POSITIVE, ids=[c[0] for c in POSITIVE])
def test_positive(statuses, cid, text, expected):
    _check(statuses, cid, text, expected)


@pytest.mark.parametrize("cid,text,expected", BORDERLINE, ids=[c[0] for c in BORDERLINE])
def test_borderline(statuses, cid, text, expected):
    _check(statuses, cid, text, expected)


@pytest.mark.parametrize("cid,text,expected", NEGATIVE, ids=[c[0] for c in NEGATIVE])
def test_negative(statuses, cid, text, expected):
    _check(statuses, cid, text, expected)


@pytest.mark.parametrize("cid,text,expected", CONTEXT, ids=[c[0] for c in CONTEXT])
def test_context(statuses, cid, text, expected):
    _check(statuses, cid, text, expected)


@pytest.mark.parametrize("cid,text,expected", ADVANCED, ids=[c[0] for c in ADVANCED])
def test_advanced(statuses, cid, text, expected):
    _check(statuses, cid, text, expected)
