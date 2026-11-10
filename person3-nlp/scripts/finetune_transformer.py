"""
Fine-tune an English Hugging Face encoder (BioBERT / ClinicalBERT / BERT) as the CONTEXT CLASSIFIER (PyTorch).

    # real run (needs internet for the pre-trained weights; a GPU/Colab makes it minutes instead of hours)
    pip install -r requirements-biogpt.txt
    python scripts/finetune_transformer.py --model dmis-lab/biobert-base-cased-v1.2 --epochs 4 --out models/context_bert

    # other English biomedical encoders you can try with the same command:
    #   emilyalsentzer/Bio_ClinicalBERT   (clinical notes)
    #   microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
    #   bert-base-uncased

    # offline smoke test (no downloads): builds a TINY random BERT + WordPiece vocabulary from the training sentences.
    # It proves the code path works; the resulting model is NOT useful for real text.
    python scripts/finetune_transformer.py --tiny-offline --epochs 2 --per-class 60 --out /tmp/tiny_ctx

Task: sentence classification into affirmed | negated | suspected | family_history | historical | hypothetical |
lab_measurement | unrelated   (nutricheck_nlp.training_data.LABELS), i.e. AutoModelForSequenceClassification.

Training details:
  * class-weighted cross-entropy (robust to class imbalance if you add your own data with --extra-train)
  * AdamW + linear warm-up/decay, gradient clipping, fixed seed
  * per-epoch precision / recall / F1 on a validation set, best epoch is kept
  * final report on the held-out cue phrases AND on the hand-written dev/blind sets (tests/data/*.jsonl)
  * saves model, tokenizer and metrics.json; nutricheck_nlp.context_model.TransformerContextClassifier loads the folder, and
    NUTRICHECK_CONTEXT_MODEL=<folder> makes the API use it instead of the TF-IDF model.

Train on REAL, de-identified, clinician-annotated English sentences (--extra-train file.jsonl with {"text","label"} lines)
before trusting it; split train/validation/test BY PATIENT/DOCUMENT to avoid leakage.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nutricheck_nlp.training_data import LABEL2ID, LABELS, generate_dataset  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def prf(gold: list[int], pred: list[int]) -> dict:
    """Per-class precision/recall/F1 + accuracy + macro-F1 (no sklearn dependency)."""
    out = {}
    f1s = []
    for i, name in enumerate(LABELS):
        tp = sum(1 for g, p in zip(gold, pred) if g == i and p == i)
        fp = sum(1 for g, p in zip(gold, pred) if g != i and p == i)
        fn = sum(1 for g, p in zip(gold, pred) if g == i and p != i)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        f1s.append(f1)
        out[name] = {"precision": round(pr, 3), "recall": round(rc, 3), "f1": round(f1, 3), "support": tp + fn}
    acc = sum(1 for g, p in zip(gold, pred) if g == p) / max(1, len(gold))
    return {"n": len(gold), "accuracy": round(acc, 4), "macro_f1": round(sum(f1s) / len(f1s), 4), "per_class": out}


def build_tiny(texts: list[str], workdir: Path):
    """Tiny random BERT + WordPiece vocabulary trained on the sentences (offline smoke test only)."""
    from tokenizers import BertWordPieceTokenizer
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    workdir.mkdir(parents=True, exist_ok=True)
    corpus = workdir / "corpus.txt"
    corpus.write_text("\n".join(texts), encoding="utf-8")
    wp = BertWordPieceTokenizer(lowercase=True, strip_accents=True)
    wp.train([str(corpus)], vocab_size=800, min_frequency=2,
             special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    wp.save_model(str(workdir))
    tok = BertTokenizerFast(vocab_file=str(workdir / "vocab.txt"), do_lower_case=True, strip_accents=True)
    cfg = BertConfig(vocab_size=tok.vocab_size, hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=128, max_position_embeddings=128, num_labels=len(LABELS),
                     id2label=dict(enumerate(LABELS)), label2id=LABEL2ID)
    return tok, BertForSequenceClassification(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dmis-lab/biobert-base-cased-v1.2")
    ap.add_argument("--tiny-offline", action="store_true", help="no downloads: tiny random model (smoke test only)")
    ap.add_argument("--per-class", type=int, default=900)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra-train", default=None, help="JSONL with your own labelled sentences")
    ap.add_argument("--holdout-eval", action="store_true",
                    help="train only on 3/4 of the cue phrases and evaluate on the unseen 1/4 (generalisation experiment)")
    ap.add_argument("--out", default=str(ROOT / "models" / "context_bert"))
    a = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    split = "train" if a.holdout_eval else "all"
    train = generate_dataset(a.per_class, seed=a.seed, split=split)
    val = generate_dataset(max(60, a.per_class // 5), seed=a.seed + 500, split="test" if a.holdout_eval else "all")
    rows = [(e.text, e.label) for e in train]
    if a.extra_train:
        rows += [(r["text"], r["label"]) for r in read_jsonl(Path(a.extra_train))]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.tiny_offline:
        tok, model = build_tiny([t for t, _ in rows], out / "_tiny_tokenizer")
    else:
        tok = AutoTokenizer.from_pretrained(a.model)
        model = AutoModelForSequenceClassification.from_pretrained(
            a.model, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=LABEL2ID)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)

    counts = [sum(1 for _, l in rows if l == name) for name in LABELS]
    weights = torch.tensor([len(rows) / (len(LABELS) * max(1, c)) for c in counts], dtype=torch.float, device=dev)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    def collate(batch):
        texts, labels = zip(*batch)
        enc = tok(list(texts), truncation=True, padding=True, max_length=a.max_len, return_tensors="pt")
        enc["labels"] = torch.tensor([LABEL2ID[l] for l in labels])
        return enc

    loader = DataLoader(rows, batch_size=a.batch, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = max(1, len(loader) * a.epochs)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    def predict(texts: list[str]) -> list[int]:
        model.eval()
        preds: list[int] = []
        with torch.no_grad():
            for i in range(0, len(texts), 64):
                enc = tok(texts[i:i + 64], truncation=True, padding=True, max_length=a.max_len, return_tensors="pt").to(dev)
                preds += model(**enc).logits.argmax(-1).tolist()
        return preds

    val_texts, val_gold = [e.text for e in val], [LABEL2ID[e.label] for e in val]
    best, best_state, history = -1.0, None, []
    for epoch in range(1, a.epochs + 1):
        model.train()
        total = 0.0
        for batch in loader:
            batch = {k: v.to(dev) for k, v in batch.items()}
            labels = batch.pop("labels")
            loss = loss_fn(model(**batch).logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            total += loss.item()
        m = prf(val_gold, predict(val_texts))
        history.append({"epoch": epoch, "train_loss": round(total / len(loader), 4), "val_accuracy": m["accuracy"], "val_macro_f1": m["macro_f1"]})
        print(f"epoch {epoch}: loss={history[-1]['train_loss']}  val accuracy={m['accuracy']}  macro-F1={m['macro_f1']}")
        if m["macro_f1"] > best:
            best = m["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    metrics: dict = {"base_model": "tiny-random (smoke test)" if a.tiny_offline else a.model, "history": history,
                     "train_examples": len(rows), "validation": prf(val_gold, predict(val_texts))}
    for name in ("dev", "blind"):
        f = ROOT / "tests" / "data" / f"context_eval_{name}.jsonl"
        if f.exists():
            r = read_jsonl(f)
            metrics[f"handwritten_{name}"] = prf([LABEL2ID[x["label"]] for x in r], predict([x["text"] for x in r]))
            print(f"handwritten_{name}: accuracy={metrics[f'handwritten_{name}']['accuracy']}  macro-F1={metrics[f'handwritten_{name}']['macro_f1']}")

    model.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved to", out)
    print("use it:  NUTRICHECK_CONTEXT_MODEL=%s uvicorn nutricheck_nlp.api.app:app --app-dir src" % out)


if __name__ == "__main__":
    main()
