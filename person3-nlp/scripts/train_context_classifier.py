"""
Train + evaluate the context classifier (scikit-learn, runs offline in seconds).

    python scripts/train_context_classifier.py                 # -> models/context_clf.joblib + reports/context_classifier_metrics.json
    python scripts/train_context_classifier.py --per-class 1500 --seed 3

What is measured (all numbers go to the JSON report AND are stored inside the model file):
    template_holdout   a PROBE model trained on 3/4 of the cue phrases, tested on the unseen 1/4 (generalisation to new wording;
                       a TF-IDF model cannot understand words it never saw - that is what a pre-trained transformer is for)
    handwritten_dev    tests/data/context_eval_dev.jsonl   - sentences written independently of the grammar (used while developing)
    handwritten_blind  tests/data/context_eval_blind.jsonl - written once, never used while developing (if present)
    The dev/blind scores are for the SHIPPED model (trained on all cue phrases).

To use your OWN annotated data, add --extra-train path.jsonl (one {"text":..., "label":...} per line).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nutricheck_nlp.context_model import SklearnContextClassifier  # noqa: E402
from nutricheck_nlp.training_data import LABELS, generate_dataset  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def evaluate(model: SklearnContextClassifier, texts: list[str], gold: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    pred = [p for p, _ in model.predict(texts)]
    p, r, f, s = precision_recall_fscore_support(gold, pred, labels=LABELS, zero_division=0)
    return {
        "n": len(gold),
        "accuracy": round(float(accuracy_score(gold, pred)), 4),
        "macro_f1": round(float(f.mean()), 4),
        "per_class": {l: {"precision": round(float(p[i]), 3), "recall": round(float(r[i]), 3), "f1": round(float(f[i]), 3),
                          "support": int(s[i])} for i, l in enumerate(LABELS)},
        "confusion_matrix": {"labels": LABELS, "matrix": confusion_matrix(gold, pred, labels=LABELS).tolist()},
        "errors": [{"text": t, "gold": g, "pred": q} for t, g, q in zip(texts, gold, pred) if g != q][:25],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=900)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--C", type=float, default=20.0)
    ap.add_argument("--extra-train", default=None, help="JSONL with your own labelled sentences")
    ap.add_argument("--out", default=str(ROOT / "models" / "context_clf.joblib"))
    ap.add_argument("--report", default=str(ROOT / "reports" / "context_classifier_metrics.json"))
    a = ap.parse_args()

    # 1) HOLD-OUT EXPERIMENT: train on 3/4 of the cue phrases, test on the unseen 1/4 -> how well does it generalise?
    train = generate_dataset(a.per_class, seed=a.seed, split="train")
    extra = read_jsonl(Path(a.extra_train)) if a.extra_train else []
    ex_t, ex_l = [x["text"] for x in extra], [x["label"] for x in extra]
    probe = SklearnContextClassifier().fit([e.text for e in train] + ex_t, [e.label for e in train] + ex_l, C=a.C)
    test = generate_dataset(max(200, a.per_class // 3), seed=a.seed + 1000, split="test")
    metrics = {"trained_on": {"examples": len(train) + len(extra), "per_class": a.per_class, "seed": a.seed, "extra": bool(extra),
                              "data": "synthetic compositional sentences (see training_data.py)"},
               "template_holdout": evaluate(probe, [e.text for e in test], [e.label for e in test])}

    # 2) SHIPPED MODEL: trained on ALL cue phrases (a cue such as "Não se pode descartar" must not be held out of the product)
    full = generate_dataset(a.per_class, seed=a.seed, split="all")
    model = SklearnContextClassifier().fit([e.text for e in full] + ex_t, [e.label for e in full] + ex_l, C=a.C)
    metrics["shipped_model_trained_on"] = {"examples": len(full) + len(extra), "split": "all cue phrases"}
    for name in ("dev", "blind"):
        f = ROOT / "tests" / "data" / f"context_eval_{name}.jsonl"
        if f.exists():
            rows = read_jsonl(f)
            metrics[f"handwritten_{name}"] = evaluate(model, [r["text"] for r in rows], [r["label"] for r in rows])
    model.metrics = {k: ({kk: vv for kk, vv in v.items() if kk not in ("errors", "confusion_matrix")} if isinstance(v, dict) else v)
                     for k, v in metrics.items()}
    model.save(a.out)
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    for k, v in metrics.items():
        if k in ("trained_on", "shipped_model_trained_on"):
            continue
        print(f"{k:20} n={v['n']:4}  accuracy={v['accuracy']:.3f}  macro-F1={v['macro_f1']:.3f}")
        weak = [f"{l}(F1={d['f1']})" for l, d in v["per_class"].items() if d["f1"] < 0.9]
        if weak:
            print("   classes below 0.90 F1:", ", ".join(weak))
    print("model  ->", a.out)
    print("report ->", a.report)


if __name__ == "__main__":
    main()
