"""
Offline tests for Person 1's code: they need only OpenCV + NumPy, NOT PaddleOCR.

They cover the real image preprocessor on the real sample image and the pure-Python parts
of MedicalOCREngine (result conversion, line extraction, confidence, image validation).
The full PaddleOCR run is covered by test_ocr_engine.py when PaddleOCR is installed.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import ocr_engine  # noqa: E402  (importable without PaddleOCR since the import is guarded)
from image_preprocessor import MedicalImagePreprocessor  # noqa: E402

SAMPLE = Path(__file__).resolve().parent / "sample_images" / "sample_report.jpg"
SAMPLE_JSON = Path(__file__).resolve().parents[1] / "outputs" / "sample_report_ocr.json"
Engine = ocr_engine.MedicalOCREngine


def _bare_engine():
    """An engine instance without PaddleOCR (only used for its pure-Python helpers)."""
    eng = object.__new__(Engine)
    eng.language, eng.cpu_threads = "pt", 1
    return eng


def test_module_imports_without_paddleocr():
    assert hasattr(ocr_engine, "MedicalOCREngine")


@pytest.mark.skipif(ocr_engine.PaddleOCR is not None, reason="PaddleOCR is installed")
def test_engine_without_paddle_raises_clear_error():
    with pytest.raises(RuntimeError, match="PaddleOCR is not installed"):
        Engine()


def test_preprocessor_on_real_sample():
    img = MedicalImagePreprocessor().preprocess(SAMPLE)
    assert isinstance(img, np.ndarray) and img.ndim == 3 and img.size > 0


def test_preprocessor_rejects_missing_file(tmp_path):
    with pytest.raises(Exception):
        MedicalImagePreprocessor().preprocess(tmp_path / "nope.jpg")


def test_preprocessor_rejects_bad_width():
    with pytest.raises(ValueError):
        MedicalImagePreprocessor(target_width=0)


def test_validate_image_ok_and_errors(tmp_path):
    eng = _bare_engine()
    assert eng._validate_image(SAMPLE) == SAMPLE
    with pytest.raises(FileNotFoundError):
        eng._validate_image(tmp_path / "missing.jpg")
    bad_ext = tmp_path / "x.gif"
    bad_ext.write_bytes(b"GIF89a")
    with pytest.raises(ValueError, match="Unsupported"):
        eng._validate_image(bad_ext)
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="could not read"):
        eng._validate_image(corrupt)
    with pytest.raises(ValueError, match="not a file"):
        eng._validate_image(tmp_path)


def test_extract_lines_and_confidence_from_paddle3_shape():
    data = {"res": {"rec_texts": ["Ferritina: 8 ng/mL", "  ", "Vitamina D: 18 ng/mL"],
                    "rec_scores": [0.9, 0.5, 0.8], "rec_boxes": [np.array([1, 2, 3, 4]), None, [5, 6, 7, 8]]}}
    lines = Engine._extract_lines(data)
    assert [l["text"] for l in lines] == ["Ferritina: 8 ng/mL", "Vitamina D: 18 ng/mL"]
    assert lines[0]["box"] == [1, 2, 3, 4]
    assert Engine._calculate_average_confidence(lines) == pytest.approx(0.85)
    assert Engine._calculate_average_confidence([]) is None


def test_extract_lines_handles_garbage():
    assert Engine._extract_lines({"res": "oops"}) == []
    assert Engine._extract_lines({"res": {"rec_texts": None, "rec_scores": None, "rec_boxes": None}}) == []


def test_result_to_dict_variants():
    class WithJson:
        json = {"res": {"rec_texts": ["a"]}}

    class WithJsonStr:
        json = json.dumps({"res": {"rec_texts": ["b"]}})

    class WithBadJson:
        json = "{not json"

    assert Engine._result_to_dict(WithJson())["res"]["rec_texts"] == ["a"]
    assert Engine._result_to_dict(WithJsonStr())["res"]["rec_texts"] == ["b"]
    assert "raw_result" in Engine._result_to_dict(WithBadJson())
    assert Engine._result_to_dict({"x": 1}) == {"x": 1}
    assert "raw_result" in Engine._result_to_dict(42)


def test_save_json_roundtrip(tmp_path):
    out = Engine.save_json({"text": "Vitamina B12: 150 pg/mL", "ok": True}, tmp_path / "sub" / "r.json")
    assert json.loads(out.read_text(encoding="utf-8"))["text"].startswith("Vitamina B12")


def test_shipped_sample_output_is_consistent():
    data = json.loads(SAMPLE_JSON.read_text(encoding="utf-8"))
    assert data["success"] and data["line_count"] == len(data["lines"]) == 7
    assert "\n".join(l["text"] for l in data["lines"]) == data["text"]
