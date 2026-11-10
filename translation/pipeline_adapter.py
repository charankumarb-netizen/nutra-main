"""Small adapter for passing OCR output into the translation component."""
from .translator import MedicalReportTranslator, TranslationResult

def translate_ocr_output(ocr_result, source_language=None, translator=None):
    """Translate OCR text/dict and return a JSON-friendly result dictionary."""
    engine = translator or MedicalReportTranslator()
    result: TranslationResult = engine.translate_ocr_result(ocr_result, source_language=source_language)
    return result.to_dict()
