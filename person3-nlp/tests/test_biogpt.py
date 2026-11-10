"""BioGPT layer tested with an injected fake backend (no torch / no download needed)."""
from nutricheck_nlp import NutriCheckNLP
from nutricheck_nlp.biogpt_engine import BioGPTSecondOpinion, BioGPTUnavailable, build_english_summary


class FakeBackend:
    def __init__(self, p=0.9, text="Fake explanation."):
        self.p, self.text, self.prompts = p, text, []

    def yes_probability(self, prompt):
        self.prompts.append(prompt)
        return self.p

    def generate(self, prompt, max_new_tokens=60):
        return self.text


def _engine(p):
    return NutriCheckNLP(biogpt=BioGPTSecondOpinion(backend=FakeBackend(p)))


def test_english_summary_content(engine):
    r = engine.analyze_text("Ferritin: 8 ng/mL. No vitamin D deficiency.")
    s = build_english_summary(r)
    assert "Iron test 8.0 ng/ml is deficient" in s and "Vitamin D is low" in s


def test_english_summary_empty(engine):
    assert "No laboratory values" in build_english_summary(engine.analyze_text("Patient bem."))


def test_agreement_no_review():
    r = _engine(0.9).analyze_text("Ferritin: 8 ng/mL", use_biogpt=True)
    assert r["biogpt"]["available"] and r["biogpt"]["per_nutrient"]["iron"]["agrees"]
    assert "iron" not in r["biogpt"]["disagreements"]


def test_disagreement_flags_review_but_never_changes_status():
    r = _engine(0.01).analyze_text("Ferritin: 8 ng/mL", use_biogpt=True)
    iron = next(n for n in r["nutrients"] if n["key"] == "iron")
    assert iron["status"] == "deficient"            # rules stay in charge
    assert "iron" in r["biogpt"]["disagreements"] and r["needs_human_review"]


def test_disagreement_on_not_deficient():
    r = _engine(0.99).analyze_text("Ferritin: 90 ng/mL", use_biogpt=True)
    assert "iron" in r["biogpt"]["disagreements"]
    assert next(n for n in r["nutrients"] if n["key"] == "iron")["status"] == "not_deficient"


def test_explain_included():
    r = _engine(0.5).analyze_text("Ferritin: 8 ng/mL", use_biogpt=True, explain=True)
    assert r["biogpt"]["explanation"] == "Fake explanation." and "explanation_warning" in r["biogpt"]


def test_prompt_is_english_and_names_nutrient():
    be = FakeBackend()
    NutriCheckNLP(biogpt=BioGPTSecondOpinion(backend=be)).analyze_text("Ferritin: 8 ng/mL", use_biogpt=True)
    assert any("Does the patient have iron deficiency?" in p for p in be.prompts)
    assert len(be.prompts) == 5


def test_probability_is_clamped():
    r = _engine(7.0).analyze_text("Ferritin: 8 ng/mL", use_biogpt=True)
    assert r["biogpt"]["per_nutrient"]["iron"]["p_yes"] == 1.0


def test_unavailable_backend_degrades(monkeypatch):
    import nutricheck_nlp.biogpt_engine as be

    def boom(*a, **k):
        raise BioGPTUnavailable("no torch")
    monkeypatch.setattr(be, "TorchBioGPTBackend", boom)
    r = NutriCheckNLP().analyze_text("Ferritin: 8 ng/mL", use_biogpt=True)
    assert r["biogpt"]["available"] is False
    assert any("BioGPT is unavailable" in w for w in r["warnings"])
    assert next(n for n in r["nutrients"] if n["key"] == "iron")["status"] == "deficient"


def test_default_model_name_env(monkeypatch):
    monkeypatch.setenv("NUTRICHECK_BIOGPT_MODEL", "my/model")
    assert BioGPTSecondOpinion().model_name == "my/model"
