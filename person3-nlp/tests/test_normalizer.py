import pytest

from nutricheck_nlp.text_normalizer import MAX_TEXT_CHARS, normalize_text, normalize_unit, sanitize_input


@pytest.mark.parametrize("text", [
    "Relatório Médico: ÁÉÍÓÚ ãõ ç ñ", "µg/dL μg/dL", "a\u00a0b\tc\r\nd", "ﬁ ligature ﬂ", "日本語 ✓ 😀", "", "x" * 1000,
    "Vitamin\u200bD", "ǅ ǆ ŉ ﬃ",
])
def test_length_preserved(text):
    assert len(normalize_text(text)) == len(text)


def test_accents_and_case():
    assert normalize_text("Relatório MÉDICO Cálcio") == "relatorio medico calcio"


def test_micro_sign_variants():
    assert normalize_text("µg") == "ug" and normalize_text("μg") == "ug"


def test_whitespace_and_dashes():
    assert normalize_text("a\u00a0b\u2013c") == "a b-c"


def test_newline_kept():
    assert "\n" in normalize_text("a\nb")


@pytest.mark.parametrize("raw,expected", [
    ("ng/mL", "ng/ml"), ("µg / dL", "ug/dl"), ("mcg/dL", "ug/dl"), ("ng/m1", "ng/ml"), ("g/d1", "g/dl"),
    ("%", "%"), (None, None), ("PG/ML", "pg/ml"),
])
def test_normalize_unit(raw, expected):
    assert normalize_unit(raw) == expected


def test_type_errors():
    with pytest.raises(TypeError):
        normalize_text(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        sanitize_input(123)  # type: ignore[arg-type]


def test_sanitize_clips():
    assert len(sanitize_input("a" * (MAX_TEXT_CHARS + 10))) == MAX_TEXT_CHARS
