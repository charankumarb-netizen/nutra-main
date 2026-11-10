"""
Run the test suites of ALL three parts of NutriCheck and print one summary.

    python run_all_tests.py            # from the project root

Person 1 (OCR) : person1-ocr/tests      (the PaddleOCR run itself is skipped when PaddleOCR is not installed)
Person 2 (data): person2-dataset/tests  (checks against the real MTSamples CSV are skipped when it is not in data/raw/)
Person 3 (NLP) : person3-nlp/tests      (rules, documents, classifier, API, Streamlit, deploy; torch tests skip without torch)
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUITES = [("Person 1 - OCR", "person1-ocr"), ("Person 2 - dataset", "person2-dataset"), ("Person 3 - NLP", "person3-nlp")]


def main() -> int:
    worst = 0
    rows = []
    for title, folder in SUITES:
        proc = subprocess.run([sys.executable, "-m", "pytest", "-o", "addopts=", "-q", "-p", "no:cacheprovider"],
                              cwd=ROOT / folder, capture_output=True, text=True)
        tail = (proc.stdout.strip().splitlines() or ["(no output)"])[-1]
        m = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|skipped|error|errors)", tail)}
        rows.append((title, m, proc.returncode))
        if proc.returncode not in (0, 5):
            worst = 1
            print(proc.stdout[-3000:], proc.stderr[-1500:])
    print("\n=== NutriCheck test summary ===")
    for title, m, rc in rows:
        print(f"{title:20} passed={m.get('passed', 0):4}  failed={m.get('failed', 0) + m.get('error', 0) + m.get('errors', 0):3}  "
              f"skipped={m.get('skipped', 0):3}   {'OK' if rc in (0, 5) else 'FAILED'}")
    print("\nALL GREEN" if worst == 0 else "\nSOME TESTS FAILED")
    return worst


if __name__ == "__main__":
    sys.exit(main())
