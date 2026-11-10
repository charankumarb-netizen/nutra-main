"""
Tests for Person 2's validator and statistics classes on small generated CSV files.

The original tests (test_dataset_py.py) check the exact public MTSamples file (4999 rows, 33 missing
transcriptions ...) and are skipped when that file is not present. These tests exercise the same
code with known, controlled inputs so the logic is verified on every machine.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_validation_py import MedicalDatasetValidator, find_dataset  # noqa: E402
from dataset_statistics_py import MedicalDatasetStatistics  # noqa: E402

COLS = ["description", "medical_specialty", "sample_name", "transcription", "keywords"]


def _csv(tmp_path, rows, name="medical_transcriptions.csv", columns=COLS):
    path = tmp_path / name
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return path


GOOD = [
    ["d1", "Hematologia", "s1", "Ferritina: 8 ng/mL. Anemia ferropriva.", "ferro"],
    ["d2", "Endocrinologia", "s2", "Vitamina D: 45 ng/mL.", "vitamina d"],
    ["d3", "Clínica", "s3", "Sem deficiência de vitamina B12.", "b12"],
]


def test_valid_dataset(tmp_path):
    rep = MedicalDatasetValidator(_csv(tmp_path, GOOD)).validate()
    assert rep["valid"] is True
    assert rep["rows"] == 3 and rep["columns"] == 5
    assert rep["duplicate_rows"] == 0
    assert rep["column_validation"]["missing_columns"] == []
    assert all(v == 0 for v in rep["missing_values"].values())


def test_missing_columns_detected(tmp_path):
    path = tmp_path / "x.csv"
    pd.DataFrame({"description": ["a"], "keywords": ["b"]}).to_csv(path, index=False)
    rep = MedicalDatasetValidator(path).validate()
    assert rep["valid"] is False
    assert "transcription" in rep["column_validation"]["missing_columns"]


def test_duplicates_counted(tmp_path):
    rep = MedicalDatasetValidator(_csv(tmp_path, GOOD + [GOOD[0]])).validate()
    assert rep["duplicate_rows"] == 1


def test_missing_and_empty_values_reported_but_not_fatal(tmp_path):
    rows = GOOD + [["d4", "Clínica", "s4", None, "k"], ["d5", "Clínica", "s5", "   ", None]]
    rep = MedicalDatasetValidator(_csv(tmp_path, rows)).validate()
    assert rep["missing_values"]["transcription"] >= 1
    assert rep["empty_text_fields"]["transcription"] == 2          # None -> "" and "   " -> ""
    assert rep["valid"] is True                                     # documented behaviour: reported, not rejected


def test_loader_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        MedicalDatasetValidator(tmp_path / "nope.csv").load()
    with pytest.raises(ValueError):
        MedicalDatasetValidator(tmp_path).load()                    # a directory
    txt = tmp_path / "data.txt"
    txt.write_text("a,b")
    with pytest.raises(ValueError, match="CSV"):
        MedicalDatasetValidator(txt).load()


def test_statistics_report(tmp_path):
    rep = MedicalDatasetStatistics(_csv(tmp_path, GOOD + [GOOD[1][:0] + ["d4", "Clínica", "s4", "um dois três quatro", "k"]])).generate_report()
    assert rep["basic_statistics"]["rows"] == 4
    assert rep["specialty_distribution"]["Clínica"] == 2
    ts = rep["transcription_statistics"]
    assert ts["minimum_words"] == 4 and ts["maximum_words"] >= 4 and ts["average_characters"] > 0


def test_statistics_requires_columns(tmp_path):
    path = tmp_path / "x.csv"
    pd.DataFrame({"foo": [1]}).to_csv(path, index=False)
    stats = MedicalDatasetStatistics(path)
    df = stats.load()
    with pytest.raises(ValueError):
        stats.specialty_distribution(df)
    with pytest.raises(ValueError):
        stats.transcription_statistics(df)


def test_find_dataset_env_and_cwd(tmp_path, monkeypatch):
    csv = _csv(tmp_path, GOOD, name="medical_transcriptions.csv.csv")   # accidental double extension
    monkeypatch.setenv("NUTRICHECK_DATASET", str(csv))
    assert find_dataset() == csv
    monkeypatch.delenv("NUTRICHECK_DATASET")
    monkeypatch.chdir(tmp_path)
    assert find_dataset().name.startswith("medical_transcriptions")


def test_find_dataset_not_found_message(tmp_path, monkeypatch):
    monkeypatch.delenv("NUTRICHECK_DATASET", raising=False)
    monkeypatch.chdir(tmp_path)
    try:
        found = find_dataset()
    except FileNotFoundError as exc:
        assert "data/raw" in str(exc)
    else:  # a real dataset is present in data/raw on this machine - also acceptable
        assert found.exists()
