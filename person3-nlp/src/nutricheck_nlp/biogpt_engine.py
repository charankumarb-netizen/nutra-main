"""
BioGPT layer (PyTorch + Hugging Face Transformers)  -  OPTIONAL "second opinion".

Design decision (important):
    The deterministic rule engine is the ONLY component that sets the final
    status.  BioGPT (an English biomedical generative LM) is used to
        1. give a probabilistic second opinion  P(yes | "does the patient have X deficiency?")
        2. write a short English explanation
    and may only RAISE a review flag when it strongly disagrees with the rules.
    Reason: generative LMs can hallucinate; a medical screening tool must be
    reproducible and auditable.

BioGPT was pre-trained on English PubMed text.  It never sees the raw report: the rule engine first condenses the
note into a structured evidence summary built ONLY from extracted fields, which is what BioGPT sees (this also means
instruction-like text inside a document cannot steer the model).

The model is loaded lazily (first use) so the API/tests start instantly and run
without torch installed.  A custom backend can be injected for testing.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Protocol

from .lexicon import TARGETS

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "microsoft/biogpt"


class BioGPTUnavailable(RuntimeError):
    """Raised when torch/transformers/model weights are not available."""


class Backend(Protocol):
    def yes_probability(self, prompt: str) -> float: ...
    def generate(self, prompt: str, max_new_tokens: int = 60) -> str: ...


class TorchBioGPTBackend:
    """Real backend: Hugging Face causal LM on PyTorch."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None) -> None:
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover - depends on environment
            raise BioGPTUnavailable(
                "torch/transformers not installed. Run: pip install -r requirements-biogpt.txt"
            ) from exc
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading %s on %s ...", model_name, self.device)
        try:  # pragma: no cover - needs network/weights
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device).eval()
        except Exception as exc:  # pragma: no cover
            raise BioGPTUnavailable(f"Could not load model '{model_name}': {exc}") from exc

    def _continuation_logprob(self, prompt: str, continuation: str) -> float:
        """log P(continuation | prompt). Prompt and continuation are tokenised SEPARATELY and concatenated, so a BPE merge
        across the boundary can never misalign the slice that is scored (the old version tokenised prompt+continuation)."""
        torch = self.torch
        p_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0]
        c_ids = self.tokenizer(continuation, add_special_tokens=False, return_tensors="pt").input_ids[0]
        if len(c_ids) == 0:
            raise ValueError("empty continuation")
        limit = int(getattr(self.model.config, "max_position_embeddings", 1024))
        keep = max(1, limit - len(c_ids))
        if len(p_ids) > keep:                       # too long for the model: keep the END of the prompt (the question)
            p_ids = p_ids[-keep:]
        full = torch.cat([p_ids, c_ids]).unsqueeze(0).to(self.device)
        n_prompt = len(p_ids)
        with torch.no_grad():
            logits = self.model(full).logits[0]
        logprobs = torch.log_softmax(logits[:-1].float(), dim=-1)
        targets = full[0, 1:]
        picked = logprobs.gather(1, targets.unsqueeze(1)).squeeze(1)
        return float(picked[n_prompt - 1:].sum())

    def yes_probability(self, prompt: str) -> float:
        lp_yes = self._continuation_logprob(prompt, " yes")
        lp_no = self._continuation_logprob(prompt, " no")
        m = max(lp_yes, lp_no)
        ey, en = math.exp(lp_yes - m), math.exp(lp_no - m)
        return ey / (ey + en)

    def generate(self, prompt: str, max_new_tokens: int = 60) -> str:
        limit = int(getattr(self.model.config, "max_position_embeddings", 1024))
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=max(8, limit - max_new_tokens)).to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return text.strip()


# ---------------------------------------------------------------------------


def build_english_summary(result: dict) -> str:
    """Structured ENGLISH evidence summary built from the rule-engine output."""
    lines: list[str] = []
    for lab in result.get("labs", []):
        if not lab.get("usable"):
            continue
        t = TARGETS[lab["target"]]
        cmp_ = lab.get("comparator") or ""
        lines.append(f"{t.name} test {cmp_}{lab['value']} {lab['unit']} is {lab['status']}.")
    for m in result.get("mentions", []):
        if m["stance"] in ("ignored",):
            continue
        t = TARGETS[m["target"]]
        state = {"low": "is low", "normal": "is normal", "high": "is high"}[m["polarity"]]
        lines.append(f"Clinical note ({m['assertion']}): {t.name} {state}.")
    if not lines:
        lines.append("No laboratory values or explicit statements were found.")
    return " ".join(lines)


class BioGPTSecondOpinion:
    def __init__(self, model_name: str | None = None, device: str | None = None, backend: Backend | None = None):
        self.model_name = model_name or os.getenv("NUTRICHECK_BIOGPT_MODEL", DEFAULT_MODEL)
        self.device = device
        self._backend = backend
        self._load_error: str | None = None

    # ------------------------------------------------------------------
    def _ensure(self) -> Backend:
        if self._backend is None:
            self._backend = TorchBioGPTBackend(self.model_name, self.device)
        return self._backend

    def is_available(self) -> tuple[bool, str | None]:
        try:
            self._ensure()
            return True, None
        except BioGPTUnavailable as exc:
            self._load_error = str(exc)
            return False, self._load_error

    # ------------------------------------------------------------------
    def review(self, result: dict, explain: bool = False) -> dict:
        ok, err = self.is_available()
        if not ok:
            return {"available": False, "model": self.model_name, "reason": err, "per_nutrient": {}}
        backend = self._ensure()
        summary = build_english_summary(result)
        per: dict[str, dict[str, Any]] = {}
        disagreements: list[str] = []
        for n in result["nutrients"]:
            t = TARGETS[n["key"]]
            prompt = f"{summary}\nQuestion: Does the patient have {t.name.lower()} deficiency?\nAnswer:"
            p = float(backend.yes_probability(prompt))
            p = min(1.0, max(0.0, p))
            status = n["status"]
            agrees = True
            if status == "deficient" and p < 0.25:
                agrees = False
            if status == "not_deficient" and p > 0.75:
                agrees = False
            if not agrees:
                disagreements.append(n["key"])
            per[n["key"]] = {"p_yes": round(p, 4), "rule_status": status, "agrees": agrees}
        out: dict[str, Any] = {
            "available": True, "model": self.model_name, "english_summary": summary,
            "per_nutrient": per, "disagreements": disagreements,
            "note": "Second opinion only; the rule engine decides the final status.",
        }
        if explain:
            out["explanation"] = backend.generate(
                f"{summary}\nBrief clinical interpretation:", max_new_tokens=60
            )
            out["explanation_warning"] = "AI-generated text; may be inaccurate. Not used for decisions."
        return out
