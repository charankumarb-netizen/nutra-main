"""Integration with Person 1 (OCR output) and Person 2 (dataset validator)."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

FIXTURES = Path(__file__).parent / "fixtures"
P2_SRC = Path(__file__).resolve().parents[2] / "person2-dataset" / "src"


def test_person1_real_ocr_output(engine):
    ocr = json.loads((FIXTURES / "person1_sample_report_ocr.json").read_text(encoding="utf-8"))
    r = engine.analyze_ocr_result(ocr)
    st = {n["key"]: n["status"] for n in r["nutrients"] + r["findings"]}
    assert st["vitamin_d"] == "deficient" and st["vitamin_b12"] == "deficient" and st["anemia"] == "deficient"
    assert st["iron"] == "not_assessed"
    assert r["ocr"]["average_confidence"] > 0.99 and not r["needs_human_review"]
    assert r["language"] == "en"


def test_person1_failed_ocr_handled(engine):
    r = engine.analyze_ocr_result({"success": False, "error": "boom", "text": ""})
    assert r["needs_human_review"] and any("OCR was not successful" in w for w in r["warnings"])
    assert all(n["status"] == "not_assessed" for n in r["nutrients"])


@pytest.mark.parametrize("bad", [None, [], "text", 5])
def test_person1_garbage_input(engine, bad):
    r = engine.analyze_ocr_result(bad)  # type: ignore[arg-type]
    assert r["needs_human_review"]


def test_person1_low_confidence_propagates(engine):
    r = engine.analyze_ocr_result({"success": True, "text": "Ferritin: 8 ng/mL", "average_confidence": 0.5,
                                   "line_count": 1, "image": "x.jpg"})
    assert r["needs_human_review"]


def test_person1_noisy_ocr_variants(engine):
    # typical OCR damage: swapped separators, spaces in units
    text = "MEDICAL REPORT\nHemoglobin : 10.2 g / dL\nVitamin D : 18 ng / mL\nVitamin B 12 : 150 pg / mL"
    st = {n["key"]: n["status"] for n in engine.analyze_text(text)["nutrients"]}
    assert st["vitamin_d"] == "deficient" and st["vitamin_b12"] == "deficient"


def test_person2_validator_and_pipeline_on_csv(engine, tmp_path):
    sys.path.insert(0, str(P2_SRC))
    from data_validation_py import MedicalDatasetValidator  # Person 2 module

    csv = tmp_path / "medical_transcriptions.csv"
    pd.DataFrame({
        "description": ["a", "b", "c"],
        "medical_specialty": ["Hematology", "Endocrinology", "Clinic"],
        "sample_name": ["s1", "s2", "s3"],
        "transcription": ["Ferritin: 8 ng/mL. Iron deficiency anemia.", "Vitamin D: 45 ng/mL.", "No B12 deficiency."],
        "keywords": ["x", "y", "z"],
    }).to_csv(csv, index=False)
    report = MedicalDatasetValidator(csv).validate()
    assert report["valid"] and report["rows"] == 3 and report["duplicate_rows"] == 0

    df = pd.read_csv(csv)
    out = [{n["key"]: n["status"] for n in engine.analyze_text(t)["nutrients"]} for t in df["transcription"]]
    assert out[0]["iron"] == "deficient" and out[1]["vitamin_d"] == "not_deficient" and out[2]["vitamin_b12"] == "not_deficient"


def test_person2_validator_rejects_bad_dataset(tmp_path):
    sys.path.insert(0, str(P2_SRC))
    from data_validation_py import MedicalDatasetValidator
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"foo": [1]}).to_csv(bad, index=False)
    assert MedicalDatasetValidator(bad).validate()["valid"] is False
    with pytest.raises(FileNotFoundError):
        MedicalDatasetValidator(tmp_path / "missing.csv").validate()
