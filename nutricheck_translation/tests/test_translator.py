import unittest
from translation.translator import MedicalReportTranslator

class TranslatorTests(unittest.TestCase):
    def setUp(self):
        self.translator = MedicalReportTranslator()

    def test_english_passes_through(self):
        text = "Hemoglobin: 10 g/dL"
        result = self.translator.translate_text(text, source_language="en")
        self.assertEqual(result.translated_text, text)
        self.assertEqual(result.status, "passed_through")

    def test_unsupported_language_stops(self):
        result = self.translator.translate_text("こんにちは", source_language="ja")
        self.assertEqual(result.status, "unsupported_language")
        self.assertEqual(result.translated_text, "")

    def test_empty_text(self):
        result = self.translator.translate_text("")
        self.assertEqual(result.status, "empty")

if __name__ == "__main__":
    unittest.main()
