"""
Trained context classifier  (Layer 2 - "fine-tuned encoder" slot of the architecture).

It reads ONE sentence and says what kind of statement it is:
    affirmed | negated | suspected | family_history | historical | hypothetical | lab_measurement | unrelated

Two interchangeable back-ends share the same interface (`predict(sentences) -> [(label, probability)]`):

    SklearnContextClassifier        TF-IDF (characters + words) + logistic regression. Trains in seconds, needs no
                                    GPU and no downloads: this is the model shipped in models/ and used by default.
    TransformerContextClassifier    A Hugging Face encoder (e.g. dmis-lab/biobert-base-cased-v1.2) fine-tuned with
                                    scripts/finetune_transformer.py. Same interface, drop-in replacement.

ROLE IN THE SYSTEM: it is a VERIFIER. The rule engine decides; if the trained model strongly disagrees about the
context of a statement, the pipeline raises `needs_human_review` (and, for text-only decisions, status `conflicting`).
It never silently overrides evidence.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Sequence

from .text_normalizer import normalize_text
from .training_data import LABELS

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "context_clf.joblib"


def prepare(text: str) -> str:
    """Model input: accent-free lower-case text with every number replaced by a token."""
    t = normalize_text(text)
    t = re.sub(r"\d+(?:[.,]\d+)?", " NUM ", t)
    return re.sub(r"\s+", " ", t).strip()


class SklearnContextClassifier:
    name = "tfidf-logreg"

    def __init__(self, pipeline: Any = None, labels: Sequence[str] | None = None, metrics: dict | None = None) -> None:
        self.pipeline = pipeline
        self.labels = list(labels) if labels is not None else list(LABELS)
        self.metrics = metrics or {}

    # -- training ----------------------------------------------------------------------------------
    @classmethod
    def build_pipeline(cls, C: float = 20.0):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion, Pipeline

        feats = FeatureUnion([
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)),
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 3), sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")),
        ])
        return Pipeline([("features", feats),
                         ("clf", LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))])

    def fit(self, texts: Sequence[str], labels: Sequence[str], C: float = 20.0) -> "SklearnContextClassifier":
        self.pipeline = self.build_pipeline(C)
        self.pipeline.fit([prepare(t) for t in texts], list(labels))
        self.labels = list(self.pipeline.classes_)
        return self

    # -- inference ---------------------------------------------------------------------------------
    def predict_proba(self, sentences: Sequence[str]):
        if self.pipeline is None:
            raise RuntimeError("model is not trained")
        return self.pipeline.predict_proba([prepare(s) for s in sentences])

    def predict(self, sentences: Sequence[str]) -> list[tuple[str, float]]:
        if not sentences:
            return []
        proba = self.predict_proba(sentences)
        out = []
        for row in proba:
            j = int(row.argmax())
            out.append((self.labels[j], float(row[j])))
        return out

    # -- persistence -------------------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        import joblib
        import sklearn

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "labels": self.labels, "metrics": self.metrics,
                     "sklearn_version": sklearn.__version__, "format": 1}, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SklearnContextClassifier":
        """SECURITY: joblib files are pickles - only load files you trained yourself or received from a trusted source."""
        import joblib

        blob = joblib.load(path)
        if not isinstance(blob, dict) or "pipeline" not in blob:
            raise ValueError("not a NutriCheck context-model file")
        return cls(blob["pipeline"], blob.get("labels"), blob.get("metrics"))


class TransformerContextClassifier:
    """Hugging Face sequence-classification model saved by scripts/finetune_transformer.py."""

    def __init__(self, model_dir: str | Path, device: str | None = None, max_length: int = 128) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except Exception as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("torch/transformers not installed: pip install -r requirements-biogpt.txt") from exc
        self.torch = torch
        self.model_dir = str(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir).to(self.device).eval()
        id2label = self.model.config.id2label
        self.labels = [id2label[i] for i in sorted(id2label)]
        self.name = f"transformer:{Path(self.model_dir).name}"
        metrics_file = Path(self.model_dir) / "metrics.json"
        self.metrics = json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file.exists() else {}

    def predict(self, sentences: Sequence[str], batch_size: int = 32) -> list[tuple[str, float]]:
        torch = self.torch
        out: list[tuple[str, float]] = []
        sentences = list(sentences)
        for i in range(0, len(sentences), batch_size):
            chunk = sentences[i:i + batch_size]
            enc = self.tokenizer(chunk, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt").to(self.device)
            with torch.no_grad():
                proba = torch.softmax(self.model(**enc).logits.float(), dim=-1).cpu()
            for row in proba:
                j = int(row.argmax())
                out.append((self.labels[j], float(row[j])))
        return out


def load_default_classifier(path: str | Path | None = None):
    """
    Load the shipped classifier (or the one named by NUTRICHECK_CONTEXT_MODEL). Returns None when it is
    missing/unreadable so the API keeps working with the rule engine alone.

    A directory is treated as a fine-tuned transformer; a file as a scikit-learn model.
    """
    target = Path(path or os.getenv("NUTRICHECK_CONTEXT_MODEL") or DEFAULT_MODEL_PATH)
    try:
        if target.is_dir():
            return TransformerContextClassifier(target)
        if target.is_file():
            return SklearnContextClassifier.load(target)
    except Exception:  # corrupt file / version mismatch / torch missing: degrade gracefully
        return None
    return None
