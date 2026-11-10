"""Run the real Streamlit script headlessly with streamlit.testing.AppTest."""
from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


def _app():
    return st_testing.AppTest.from_file(APP, default_timeout=60)


def test_app_loads_without_exception():
    at = _app().run()
    assert not at.exception
    assert "NutriCheck" in at.title[0].value


def test_analyze_sample_text():
    at = _app().run()
    at.selectbox(key="sample").select("OCR report (Person 1 style)").run()
    at.button[0].click().run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert "Deficiency detected" in metrics["Vitamin B12"]
    assert "Deficiency detected" in metrics["Vitamin D"]


def test_empty_text_shows_error():
    at = _app().run()
    at.text_area[0].set_value("   ").run()
    at.button[0].click().run()
    assert not at.exception and len(at.error) == 1


def test_negation_sample_renders():
    at = _app().run()
    at.selectbox(key="sample").select("Negation and context").run()
    at.button[0].click().run()
    assert not at.exception
    m = {x.label: x.value for x in at.metric}
    assert "No deficiency" in m["Iron"] and "Deficiency detected" in m["Vitamin D"]
