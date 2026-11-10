
import json
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
SRC_DIR = PROJECT_DIR / "src"

sys.path.insert(0, str(SRC_DIR))

from report_parser import parse_medical_report

json_path = PROJECT_DIR / "outputs" / "real_report_ocr.json"

with open(json_path, "r", encoding="utf-8") as file:
    ocr_result = json.load(file)

parsed = parse_medical_report(ocr_result["text"])

print("\nSTRUCTURED REPORT")
print(json.dumps(parsed, indent=4, ensure_ascii=False))

assert parsed["patient_name"] == "João Silva"
assert parsed["age"] == 35
assert len(parsed["tests"]) == 5

print("\nAll parser checks passed!")
