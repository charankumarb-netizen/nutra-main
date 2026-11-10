"""
Evaluate the engine on the gold cases + N synthetic cases and write reports/evaluation.json|md

    python scripts/evaluate.py --synthetic 5000 --seed 7
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

from gold_cases import ALL_CASES  # noqa: E402
from gold_cases_v2 import ALL_V2  # noqa: E402
from nutricheck_nlp import NutriCheckNLP  # noqa: E402
from nutricheck_nlp.synthetic import generate_cases  # noqa: E402


def run(n_syn: int, seed: int) -> dict:
    eng = NutriCheckNLP()
    stats = defaultdict(lambda: defaultdict(int))
    fails = []

    def check(src, text, expected, **kw):
        r = eng.analyze_text(text, **kw)
        got = {n["key"]: n["status"] for n in r["nutrients"] + r["findings"]}
        for k, exp in expected.items():
            s = stats[src]
            s["total"] += 1
            if got[k] == exp:
                s["correct"] += 1
            else:
                fails.append({"source": src, "text": text, "target": k, "expected": exp, "got": got[k]})
            # deficiency detection confusion (positive class = 'deficient')
            if k != "anemia":
                p, g = got[k] == "deficient", exp == "deficient"
                s["tp"] += p and g
                s["fp"] += p and not g
                s["fn"] += (not p) and g

    for case in list(ALL_CASES) + list(ALL_V2):          # 177 + 193 hand-written cases
        check("gold", case[1], case[2], **(case[3] if len(case) > 3 else {}))
    for c in generate_cases(n_syn, seed):
        check("synthetic", c.text, c.expected)

    report = {}
    for src, s in stats.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        report[src] = {
            "targets_checked": s["total"], "accuracy": round(s["correct"] / s["total"], 6),
            "deficiency_precision": round(tp / (tp + fp), 6) if tp + fp else None,
            "deficiency_recall": round(tp / (tp + fn), 6) if tp + fn else None,
        }
    return {"summary": report, "failures": fails[:50], "n_failures": len(fails)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rep = run(a.synthetic, a.seed)
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "evaluation.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Evaluation report", ""]
    for src, m in rep["summary"].items():
        lines.append(f"- **{src}**: accuracy {m['accuracy']:.4%} over {m['targets_checked']} labels; "
                     f"deficiency precision {m['deficiency_precision']}, recall {m['deficiency_recall']}")
    lines.append(f"\nFailures: {rep['n_failures']}")
    (out / "evaluation.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    sys.exit(1 if rep["n_failures"] else 0)
