"""
Run the engine over Person 2's cleaned CSV (column 'transcription') and summarise.
The public MTSamples data is English, which is exactly what the engine reads: no translation step is needed.

    python scripts/scan_dataset.py data/raw/medical_transcriptions.csv --limit 500
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nutricheck_nlp import NutriCheckNLP  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--column", default="transcription")
ap.add_argument("--limit", type=int, default=500)
ap.add_argument("--out", default=str(ROOT / "reports" / "dataset_scan.csv"))
a = ap.parse_args()

df = pd.read_csv(a.csv).head(a.limit)
eng = NutriCheckNLP()
rows, langs = [], Counter()
for t in df[a.column].fillna("").astype(str):
    r = eng.analyze_text(t)
    langs[r["language"]] += 1
    rows.append({n["key"]: n["status"] for n in r["nutrients"] + r["findings"]} | {"review": r["needs_human_review"]})
print("Detected languages:", dict(langs))
unsupported = sum(v for k, v in langs.items() if k not in ("en", "unknown"))
if unsupported > len(df) / 2:
    print("WARNING: most records are not English; the engine refuses to interpret them (status not_assessed).")
out = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
out.to_csv(a.out, index=False)
for k in ["iron", "vitamin_d", "vitamin_b12", "calcium", "folate", "anemia"]:
    print(k, dict(Counter(out[k])))
