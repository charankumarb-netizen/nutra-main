"""
Real PyTorch / Hugging Face code paths, exercised OFFLINE with tiny randomly-initialised models.

They prove that the code RUNS end to end (training loop, saving/loading, inference, BioGPT scoring and generation, integration
in the pipeline). They say nothing about clinical quality: for that, fine-tune a real model (dmis-lab/biobert-base-cased-v1.2,
pucpr/biobertpt-all, microsoft/biogpt) as described in docs/PERSON3_NLP_GUIDE.md.
Skipped automatically when torch / transformers are not installed.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from nutricheck_nlp import NutriCheckNLP  # noqa: E402
from nutricheck_nlp.biogpt_engine import BioGPTSecondOpinion, TorchBioGPTBackend  # noqa: E402
from nutricheck_nlp.context_model import TransformerContextClassifier, load_default_classifier  # noqa: E402
from nutricheck_nlp.training_data import LABELS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tiny_bert(tmp_path_factory):
    out = tmp_path_factory.mktemp("tiny_bert")
    run = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "finetune_transformer.py"), "--tiny-offline", "--per-class", "40",
         "--epochs", "2", "--batch", "16", "--out", str(out)],
        capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stdout[-1500:] + run.stderr[-1500:]
    return out, run.stdout


def test_finetune_script_trains_saves_and_reports(tiny_bert):
    out, stdout = tiny_bert
    assert "epoch 1:" in stdout and "epoch 2:" in stdout and "handwritten_blind" in stdout
    for f in ("config.json", "metrics.json", "tokenizer.json"):
        assert (out / f).exists(), f
    assert (out / "model.safetensors").exists() or (out / "pytorch_model.bin").exists()
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert len(metrics["history"]) == 2 and set(metrics["validation"]["per_class"]) == set(LABELS)
    assert metrics["handwritten_blind"]["n"] == 120 and "tiny-random" in metrics["base_model"]

def test_transformer_classifier_loads_and_predicts(tiny_bert):
    out, _ = tiny_bert
    clf = TransformerContextClassifier(out)
    assert clf.labels == LABELS and clf.name.startswith("transformer:")
    preds = clf.predict(["No iron deficiency.", "Ferritin: 8 ng/mL", "Mother with B12 deficiency."] * 20, batch_size=7)
    assert len(preds) == 60 and all(l in LABELS and 0 < p <= 1 for l, p in preds)
    assert clf.predict([]) == []
    assert clf.metrics["train_examples"] > 0


def test_loader_accepts_a_transformer_folder(tiny_bert):
    out, _ = tiny_bert
    assert isinstance(load_default_classifier(out), TransformerContextClassifier)


def test_transformer_inside_the_pipeline_is_only_a_verifier(tiny_bert):
    out, _ = tiny_bert
    eng = NutriCheckNLP(classifier=TransformerContextClassifier(out), classifier_threshold=0.0)   # 0.0 = every disagreement counts
    res = eng.analyze_text("Ferritin: 8 ng/mL. No vitamin D deficiency.")
    assert res["classifier"]["available"] and res["classifier"]["name"].startswith("transformer:")
    iron = next(n for n in res["nutrients"] if n["key"] == "iron")
    assert iron["status"] == "deficient"                                   # lab-decided: a random model cannot change it


# ------------------------------------------------------------------ BioGPT (real backend, tiny random model)
@pytest.fixture(scope="module")
def tiny_biogpt(tmp_path_factory):
    from transformers import BioGptConfig, BioGptForCausalLM, BioGptTokenizer

    d = tmp_path_factory.mktemp("tiny_biogpt")
    alphabet = list("abcdefghijklmnopqrstuvwxyz0123456789.,:;?!-()/%'\"=<>")
    vocab = ["<s>", "<pad>", "</s>", "<unk>"] + alphabet + [c + "</w>" for c in alphabet]
    (d / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(vocab)}), encoding="utf-8")
    (d / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    tok = BioGptTokenizer(str(d / "vocab.json"), str(d / "merges.txt"))
    cfg = BioGptConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                       intermediate_size=64, max_position_embeddings=256, bos_token_id=0, pad_token_id=1, eos_token_id=2)
    torch.manual_seed(0)
    BioGptForCausalLM(cfg).save_pretrained(d)
    tok.save_pretrained(d)
    return d


def test_biogpt_backend_scores_and_generates(tiny_biogpt):
    be = TorchBioGPTBackend(str(tiny_biogpt), device="cpu")
    prompt = "Question: is the iron test of 8 ng/ml low? Answer:"
    p1, p2 = be.yes_probability(prompt), be.yes_probability(prompt)
    assert 0.0 < p1 < 1.0 and p1 == pytest.approx(p2)                       # deterministic probability in (0, 1)
    assert isinstance(be.generate(prompt, max_new_tokens=8), str)


def test_biogpt_prompt_longer_than_the_model_is_truncated_not_crashing(tiny_biogpt):
    be = TorchBioGPTBackend(str(tiny_biogpt), device="cpu")
    long_prompt = "iron " * 600 + "Answer:"
    assert 0.0 < be.yes_probability(long_prompt) < 1.0
    assert isinstance(be.generate(long_prompt, max_new_tokens=8), str)


def test_biogpt_second_opinion_with_real_backend_never_changes_status(tiny_biogpt):
    text = "Ferritin: 8 ng/mL. Vitamin D: 50 ng/mL."
    base = NutriCheckNLP().analyze_text(text)
    eng = NutriCheckNLP(biogpt=BioGPTSecondOpinion(model_name=str(tiny_biogpt), device="cpu"))
    res = eng.analyze_text(text, use_biogpt=True, explain=True)
    assert res["biogpt"]["available"] and set(res["biogpt"]["per_nutrient"]) >= {"iron", "vitamin_d"}
    assert {n["key"]: n["status"] for n in res["nutrients"]} == {n["key"]: n["status"] for n in base["nutrients"]}
    assert "explanation" in res["biogpt"]


def test_models_status_reports_loaded_biogpt(tiny_biogpt):
    eng = NutriCheckNLP(biogpt=BioGPTSecondOpinion(model_name=str(tiny_biogpt), device="cpu"))
    assert eng.models_status()["biogpt"]["torch_installed"] is True
    eng.analyze_text("Ferritin: 8 ng/mL", use_biogpt=True)
    assert eng.models_status()["biogpt"]["loaded"] is True
