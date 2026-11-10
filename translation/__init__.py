"""NutriCheck Portuguese/English medical-report translation package."""
from .translator import MedicalReportTranslator, TranslationResult
from .language_detector import detect_language, LanguageDetection
from .pipeline_adapter import translate_ocr_output

__all__ = ["MedicalReportTranslator", "TranslationResult", "detect_language", "LanguageDetection", "translate_ocr_output"]
