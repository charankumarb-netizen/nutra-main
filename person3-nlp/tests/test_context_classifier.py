"""Training data, the trained context classifier, and its role as VERIFIER of the rule engine."""
import json
from pathlib import Path

import pytest

from nutricheck_nlp import NutriCheckNLP
from nutricheck_nlp.context_model import (
    DEFAULT_MODEL_PATH, SklearnContextClassifier, load_default_classifier, prepare,
)
from nutricheck_nlp.training_data import LABELS, add_noise, generate_dataset

DATA = Path(__file__).resolve().parent / "data"


def read(name):
    return [json.loads(l) for l in (DATA / name).read_text(encoding="utf-8").splitlines() if l.strip()]


@pytest.fixture(scope="module")
def clf():
    c = load_default_classifier()
    assert c is not None, "models/context_clf.joblib is missing: run scripts/train_context_classifier.py"
    return c


class Fake:
    """Deterministic stand-in classifier: returns scripted (label, probability) for every sentence."""
    name = "fake"
    labels = LABELS

    def __init__(self, label, prob=0.97):
        self.label, self.prob = label, prob

    def predict(self, sentences):
        return [(self.label, self.prob) for _ in sentences]


# ------------------------------------------------------------------ training data
def test_dataset_is_balanced_deterministic_and_complete():
    a, b = generate_dataset(40, seed=7), generate_dataset(40, seed=7)
    assert [(e.text, e.label) for e in a] == [(e.text, e.label) for e in b]
    assert {e.label for e in a} == set(LABELS)
    assert all(sum(1 for e in a if e.label == l) == 40 for l in LABELS)
    assert generate_dataset(40, seed=8)[0].text != a[0].text or generate_dataset(40, seed=8) != a


def test_train_and_test_cues_are_disjoint():
    from nutricheck_nlp.training_data import _GRAMMAR, _cues

    for label in _GRAMMAR:
        for which in (0, 1):
            tr = {c for _, c in _cues(label, which, "train")} - {""}
            te = {c for _, c in _cues(label, which, "test")} - {""}
            assert tr and te and not tr & te, (label, which, tr & te)


def test_hard_negatives_reuse_cue_words_without_nutrients():
    texts = [e.text.lower() for e in generate_dataset(300, seed=1) if e.label == "unrelated"]
    assert sum(1 for t in texts if any(w in t for w in ("suspected", "no ", "mother", "history", "risk"))) > 30
    assert not any(n in t for t in texts for n in ("ferritin", "vitamin d", "b12", "iron", "folate"))


def test_noise_changes_text_but_keeps_it_readable():
    import random

    rng = random.Random(0)
    outs = {add_noise("No vitamin B12 deficiency.", rng) for _ in range(200)}
    assert len(outs) > 4 and all(len(o) >= 20 for o in outs)


def test_prepare_normalises_numbers_and_accents():
    assert prepare("Ferritin: 8.5 ng/mL") == "ferritin: NUM ng/ml"


# ------------------------------------------------------------------ the shipped model
def test_model_file_exists_and_loads(clf):
    assert DEFAULT_MODEL_PATH.is_file() and isinstance(clf, SklearnContextClassifier)
    assert set(clf.labels) == set(LABELS)


def test_predict_interface(clf):
    out = clf.predict(["No iron deficiency.", "Ferritin: 8 ng/mL", ""])
    assert len(out) == 3 and all(isinstance(l, str) and 0 < p <= 1 for l, p in out)
    assert clf.predict([]) == []


@pytest.mark.parametrize("sentence,label", [
    ("Patient has vitamin D deficiency.", "affirmed"),
    ("There is no iron deficiency.", "negated"),
    ("Suspected vitamin B12 deficiency.", "suspected"),
    ("Mother with iron deficiency.", "family_history"),
    ("History of iron deficiency, already treated.", "historical"),
    ("Advice to prevent vitamin D deficiency.", "hypothetical"),
    ("Ferritin: 8 ng/mL", "lab_measurement"),
    ("Blood pressure 120/80 mmHg.", "unrelated"),
    ("No visual impairment.", "unrelated"),
])
def test_prototype_sentences(clf, sentence, label):
    assert clf.predict([sentence])[0][0] == label


def test_documented_quality_floor_on_handwritten_sets(clf):
    """The numbers in reports/context_classifier_metrics.json are real: accuracy on hand-written sentences must stay high."""
    for name, floor in (("context_eval_dev.jsonl", 0.90), ("context_eval_blind.jsonl", 0.85)):
        rows = read(name)
        pred = [l for l, _ in clf.predict([r["text"] for r in rows])]
        acc = sum(p == r["label"] for p, r in zip(pred, rows)) / len(rows)
        assert acc >= floor, (name, acc)


def test_high_confidence_predictions_are_more_accurate(clf):
    rows = read("context_eval_dev.jsonl") + read("context_eval_blind.jsonl")
    pred = clf.predict([r["text"] for r in rows])
    sure = [(p, r) for (p, pr), r in zip(pred, rows) if pr >= 0.85]
    assert len(sure) > 0.6 * len(rows) and sum(p == r["label"] for p, r in sure) / len(sure) >= 0.94


def test_metrics_are_embedded_in_the_model(clf):
    m = clf.metrics
    assert {"template_holdout", "handwritten_dev", "handwritten_blind"} <= set(m)
    assert m["handwritten_blind"]["n"] == 120 and m["template_holdout"]["accuracy"] < 0.95   # honest: unseen wording is hard


def test_save_load_roundtrip(tmp_path):
    data = generate_dataset(30, seed=2)
    m = SklearnContextClassifier().fit([e.text for e in data], [e.label for e in data])
    path = m.save(tmp_path / "x" / "m.joblib")
    m2 = SklearnContextClassifier.load(path)
    s = ["No iron deficiency.", "Mother with iron deficiency."]
    assert m.predict(s) == m2.predict(s)


def test_loader_degrades_gracefully(tmp_path, monkeypatch):
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"garbage")
    assert load_default_classifier(bad) is None
    assert load_default_classifier(tmp_path / "missing.joblib") is None
    import joblib
    joblib.dump({"unrelated": 1}, tmp_path / "wrong.joblib")
    assert load_default_classifier(tmp_path / "wrong.joblib") is None
    monkeypatch.setenv("NUTRICHECK_CONTEXT_MODEL", str(tmp_path / "nope"))
    assert load_default_classifier() is None


# ------------------------------------------------------------------ verifier behaviour inside the pipeline
def status(res, key="iron"):
    return next(n for n in res["nutrients"] if n["key"] == key)


def test_agreement_changes_nothing():
    eng = NutriCheckNLP(classifier=Fake("negated"))
    res = eng.analyze_text("No iron deficiency.")
    assert status(res)["status"] == "not_deficient" and not res["needs_human_review"]
    assert res["classifier"]["items"][0]["agrees"] and res["classifier"]["disagreements"] == []


def test_confident_disagreement_on_text_only_decision_gives_conflicting():
    eng = NutriCheckNLP(classifier=Fake("affirmed", 0.97))
    res = eng.analyze_text("No iron deficiency.")
    n = status(res)
    assert n["status"] == "conflicting" and n["needs_review"] and res["needs_human_review"]
    assert {"classifier_disagreement", "conflicting_evidence"} <= set(n["review_reasons"])
    assert res["classifier"]["disagreements"] == ["iron"] and res["mentions"][0]["classifier"]["model_label"] == "affirmed"


def test_low_probability_disagreement_is_ignored():
    res = NutriCheckNLP(classifier=Fake("affirmed", 0.60)).analyze_text("No iron deficiency.")
    assert status(res)["status"] == "not_deficient" and not res["needs_human_review"]


def test_lab_value_is_never_overridden_by_the_classifier():
    res = NutriCheckNLP(classifier=Fake("negated", 0.99)).analyze_text("Ferritin: 8 ng/mL. Iron deficiency.")
    n = status(res)
    assert n["status"] == "deficient" and n["lab_status"] == "deficient"        # evidence stays in charge...
    assert n["needs_review"] and "classifier_disagreement" in n["review_reasons"]   # ...but a human is asked to look


def test_flag_policy_keeps_status_and_only_requests_review():
    eng = NutriCheckNLP(classifier=Fake("affirmed"), classifier_policy="flag")
    res = eng.analyze_text("No iron deficiency.")
    assert status(res)["status"] == "not_deficient" and res["needs_human_review"]


def test_use_classifier_false_and_true_without_model(engine):
    eng = NutriCheckNLP(classifier=Fake("affirmed"))
    assert eng.analyze_text("No iron deficiency.", use_classifier=False)["classifier"] is None
    res = engine.analyze_text("No iron deficiency.", use_classifier=True)          # engine fixture has no classifier
    assert res["classifier"] is None and any("unavailable" in w for w in res["warnings"])


def test_a_crashing_classifier_never_breaks_the_analysis():
    class Boom(Fake):
        def predict(self, s):
            raise ValueError("model exploded")

    res = NutriCheckNLP(classifier=Boom("x")).analyze_text("No iron deficiency.")
    assert status(res)["status"] == "not_deficient" and any("failed" in w for w in res["warnings"])


def test_invalid_policy_rejected():
    with pytest.raises(ValueError):
        NutriCheckNLP(classifier_policy="yolo")


def test_clause_isolation_two_statements_in_one_sentence():
    """'No iron deficiency, mas com carência de B12' = two clauses; the classifier must see each one separately."""
    seen = []

    class Spy(Fake):
        def predict(self, sentences):
            seen.extend(sentences)
            return super().predict(sentences)

    NutriCheckNLP(classifier=Spy("negated")).analyze_text("No iron deficiency, but with vitamin B12 deficiency.")
    assert len(seen) == 2 and "but" not in seen[0].lower().split() and "iron" not in seen[1].lower()


def test_ignored_mentions_are_verified_but_cannot_flip_status():
    """Family history is ignored by the rules; a confident 'affirmed' from the model is flagged, status stays not_assessed."""
    res = NutriCheckNLP(classifier=Fake("affirmed")).analyze_text("Mother with iron deficiency.")
    n = status(res)
    assert n["status"] == "not_assessed" and n["needs_review"] and res["needs_human_review"]


def test_real_model_agrees_with_rules_on_clean_clinical_sentences(clf):
    """With the real trained model, clearly-worded sentences must not trigger review."""
    eng = NutriCheckNLP(classifier=clf)
    for text, key, expected in [
        ("Patient has vitamin D deficiency.", "vitamin_d", "deficient"),
        ("No iron deficiency.", "iron", "not_deficient"),
        ("Suspected vitamin B12 deficiency.", "vitamin_b12", "suspected"),
        ("Mother with iron deficiency.", "iron", "not_assessed"),
        ("Ferritin normal.", "iron", "not_deficient"),
    ]:
        res = eng.analyze_text(text)
        n = next(x for x in res["nutrients"] if x["key"] == key)
        assert n["status"] == expected, text
        assert "classifier_disagreement" not in n["review_reasons"], (text, res["classifier"]["items"])


def test_status_never_changes_for_lab_decided_cases_with_real_model(clf):
    """Property over ALL gold cases: if the status is lab-decided, the real classifier can only add review flags."""
    from gold_cases import ALL_CASES
    from gold_cases_v2 import ALL_V2

    plain, with_clf = NutriCheckNLP(), NutriCheckNLP(classifier=clf)
    for c in ALL_CASES + ALL_V2:
        kw = c[3] if len(c) > 3 else {}
        a, b = plain.analyze_text(c[1], **kw), with_clf.analyze_text(c[1], **kw)
        for x, y in zip(a["nutrients"] + a["findings"], b["nutrients"] + b["findings"]):
            if x["lab_status"] != "none":
                assert x["status"] == y["status"], (c[0], x["key"])


def test_real_model_status_flip_rate_is_small(clf):
    """Quantify the verifier's side effect on the 362 hand-written gold cases (text-only decisions can become 'conflicting')."""
    from gold_cases import ALL_CASES
    from gold_cases_v2 import ALL_V2

    plain, with_clf = NutriCheckNLP(), NutriCheckNLP(classifier=clf)
    flips = total = 0
    for c in ALL_CASES + ALL_V2:
        kw = c[3] if len(c) > 3 else {}
        a, b = plain.analyze_text(c[1], **kw), with_clf.analyze_text(c[1], **kw)
        for x, y in zip(a["nutrients"] + a["findings"], b["nutrients"] + b["findings"]):
            if x["status"] != "not_assessed":
                total += 1
                flips += x["status"] != y["status"]
    assert flips / total < 0.06, (flips, total)
