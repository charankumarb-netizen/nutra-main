"""Romance-language to English medical-report translation for NutriCheck.

Uses Helsinki-NLP/opus-mt-ROMANCE-en. Translation is an aid only; always
review it against the original OCR text and the source report.
"""
from dataclasses import dataclass, asdict
from typing import Optional
import re

from .language_detector import detect_language


@dataclass
class TranslationResult:
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    status: str
    confidence: Optional[float] = None
    warnings: tuple = ()

    def to_dict(self):
        data = asdict(self)
        data["warnings"] = list(self.warnings)
        return data


class MedicalReportTranslator:
    """Translate supported Romance languages to English; pass English through."""

    DEFAULT_MODEL = "Helsinki-NLP/opus-mt-ROMANCE-en"
    SUPPORTED_SOURCE_LANGUAGES = {
        "fr", "fr_be", "fr_ca", "fr_fr", "wa", "frp", "oc", "ca", "rm",
        "lld", "fur", "lij", "lmo", "es", "es_ar", "es_cl", "es_co",
        "es_cr", "es_do", "es_ec", "es_es", "es_gt", "es_hn", "es_mx",
        "es_ni", "es_pa", "es_pe", "es_pr", "es_sv", "es_uy", "es_ve",
        "pt", "pt_br", "pt-br", "pt-br", "pt_pt", "gl", "lad", "an",
        "mwl", "it", "it_it", "co", "nap", "scn", "vec", "sc", "ro", "la",
    }

    def __init__(self, model_name: str = DEFAULT_MODEL, device: int = -1, max_length: int = 512):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        if self._model is None:
            try:
                import torch
                from transformers import MarianMTModel, MarianTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "Translation dependencies are missing. Run: "
                    "python -m pip install -r translation/requirements.txt"
                ) from exc
            self._tokenizer = MarianTokenizer.from_pretrained(self.model_name)
            self._model = MarianMTModel.from_pretrained(self.model_name)
            if self.device >= 0 and torch.cuda.is_available():
                self._model.to(torch.device(f"cuda:{self.device}"))
            else:
                self._model.to(torch.device("cpu"))
        return self._model, self._tokenizer

    @staticmethod
    def _protect_numeric_tokens(text: str):
        pattern = re.compile(
            r"(?<!\w)(?:\d{1,4}[./-]\d{1,2}[./-]\d{1,4}|"
            r"\d+(?:[.,]\d+)?\s*(?:-|–|to)\s*\d+(?:[.,]\d+)?|"
            r"\d+(?:[.,]\d+)?\s*(?:%|mg/dL|g/dL|mmol/L|µmol/L|umol/L|"
            r"ng/mL|pg/mL|mL|min|cm|kg|mmHg|IU/L|U/L|fL|pg|mg|g|ng|µg|ug|"
            r"mcg|mmol|µmol|umol|°C)?)(?!\w)", re.IGNORECASE
        )
        tokens = []

        def replace(match):
            marker = f"ZXQNUM{len(tokens)}QXZ"
            tokens.append(match.group(0))
            return marker

        return pattern.sub(replace, text), tokens

    @staticmethod
    def _restore_numeric_tokens(text: str, tokens):
        for index, token in enumerate(tokens):
            text = text.replace(f"ZXQNUM{index}QXZ", token)
        return text

    def translate_text(self, text: str, source_language: Optional[str] = None) -> TranslationResult:
        original = text or ""
        if not original.strip():
            return TranslationResult(original, "", "unknown", "en", "empty", warnings=("OCR text is empty.",))

        detection = detect_language(original) if source_language is None else None
        source = (source_language or detection.language or "unknown").lower().replace("-", "_")
        if source in {"en", "english"}:
            return TranslationResult(original, original, "en", "en", "passed_through",
                                     confidence=(detection.confidence if detection else 1.0))
        if source not in self.SUPPORTED_SOURCE_LANGUAGES:
            return TranslationResult(
                original, "", source, "en", "unsupported_language",
                confidence=(detection.confidence if detection else None),
                warnings=(
                    "The model supports Romance-language sources such as Portuguese, Spanish, French, "
                    "Italian, Catalan and Romanian. Pass source_language explicitly when detection is uncertain.",
                ),
            )

        protected, tokens = self._protect_numeric_tokens(original)
        try:
            model, tokenizer = self._load_model()
            import torch
            inputs = tokenizer(protected, return_tensors="pt", truncation=True, max_length=self.max_length)
            model_device = next(model.parameters()).device
            inputs = {key: value.to(model_device) for key, value in inputs.items()}
            with torch.no_grad():
                generated = model.generate(**inputs, max_length=self.max_length, num_beams=4)
            translated = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
            translated = self._restore_numeric_tokens(translated, tokens)
            warnings = (
                "Review the English translation against the original report.",
                "Manually check clinical negation, abnormal/normal labels, decimal separators, dates and units.",
                "This model is not clinically validated and must not be used as the sole basis for diagnosis.",
            )
            return TranslationResult(original, translated.strip(), source, "en", "translated",
                                     confidence=(detection.confidence if detection else None), warnings=warnings)
        except Exception as exc:
            return TranslationResult(
                original, "", source, "en", "translation_error",
                warnings=(f"Translation failed: {exc}",
                          "Do not send untranslated source text to the English-only NLP stage."),
            )

    def translate_ocr_result(self, ocr_result, source_language: Optional[str] = None):
        if isinstance(ocr_result, str):
            text = ocr_result
        elif isinstance(ocr_result, dict):
            text = ocr_result.get("text") or ocr_result.get("extracted_text") or ""
            if not text and isinstance(ocr_result.get("lines"), list):
                text = "\n".join(
                    str(item.get("text", "")) if isinstance(item, dict) else str(item)
                    for item in ocr_result["lines"]
                )
        else:
            raise TypeError("ocr_result must be a string or dictionary.")
        return self.translate_text(text, source_language=source_language)
