"""
Tests for the NutriCheck OCR Engine.

Run from the NutriCheck project root:

    python -m unittest discover -s person1-ocr/tests -v
"""

import json
import sys
import unittest
from pathlib import Path


# ---------------------------------------------------------
# Make the src folder importable
# ---------------------------------------------------------

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent / "src"

sys.path.insert(0, str(SRC_DIR))

try:
    import paddleocr  # noqa: F401
except ImportError:
    raise unittest.SkipTest(
        "PaddleOCR is not installed (pip install -r requirements-ocr.txt); "
        "offline OCR-logic tests are in test_ocr_logic_offline.py"
    )

from ocr_engine import MedicalOCREngine


# ---------------------------------------------------------
# Test paths
# ---------------------------------------------------------

SAMPLE_IMAGE = (
    CURRENT_DIR
    / "sample_images"
    / "sample_report.jpg"
)

OUTPUT_DIR = (
    CURRENT_DIR.parent
    / "outputs"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "sample_report_ocr.json"
)


# ---------------------------------------------------------
# OCR Engine Tests
# ---------------------------------------------------------

class TestMedicalOCREngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Create the OCR engine once for all tests.
        """

        cls.engine = MedicalOCREngine(
            language="pt",
            cpu_threads=4,
        )

    # -----------------------------------------------------
    # Test 1: Sample image exists
    # -----------------------------------------------------

    def test_sample_image_exists(self):

        self.assertTrue(
            SAMPLE_IMAGE.exists(),
            f"Sample image not found: {SAMPLE_IMAGE}",
        )

    # -----------------------------------------------------
    # Test 2: OCR returns a dictionary
    # -----------------------------------------------------

    def test_ocr_returns_dictionary(self):

        result = self.engine.extract_text(
            SAMPLE_IMAGE,
            preprocess=True,
        )

        self.assertIsInstance(
            result,
            dict,
        )

    # -----------------------------------------------------
    # Test 3: OCR detects text
    # -----------------------------------------------------

    def test_ocr_detects_text(self):

        result = self.engine.extract_text(
            SAMPLE_IMAGE,
            preprocess=True,
        )

        self.assertGreater(
            len(result["lines"]),
            0,
            "OCR did not detect any text.",
        )

    # -----------------------------------------------------
    # Test 4: Extracted text is not empty
    # -----------------------------------------------------

    def test_extracted_text_not_empty(self):

        result = self.engine.extract_text(
            SAMPLE_IMAGE,
            preprocess=True,
        )

        self.assertTrue(
            result["text"].strip(),
            "Extracted OCR text is empty.",
        )

    # -----------------------------------------------------
    # Test 5: Confidence exists
    # -----------------------------------------------------

    def test_confidence_exists(self):

        result = self.engine.extract_text(
            SAMPLE_IMAGE,
            preprocess=True,
        )

        self.assertIsNotNone(
            result["average_confidence"],
            "Average OCR confidence was not generated.",
        )

        self.assertGreaterEqual(
            result["average_confidence"],
            0.0,
        )

        self.assertLessEqual(
            result["average_confidence"],
            1.0,
        )

    # -----------------------------------------------------
    # Test 6: Each OCR line has required fields
    # -----------------------------------------------------

    def test_line_structure(self):

        result = self.engine.extract_text(
            SAMPLE_IMAGE,
            preprocess=True,
        )

        for line in result["lines"]:

            self.assertIn(
                "text",
                line,
            )

            self.assertIn(
                "confidence",
                line,
            )

            self.assertIsInstance(
                line["text"],
                str,
            )

    # -----------------------------------------------------
    # Test 7: JSON output can be created
    # -----------------------------------------------------

    def test_json_output(self):

        result = self.engine.extract_text(
            SAMPLE_IMAGE,
            preprocess=True,
        )

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            OUTPUT_JSON,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                result,
                file,
                ensure_ascii=False,
                indent=2,
            )

        self.assertTrue(
            OUTPUT_JSON.exists(),
        )

        # Verify that the generated file is valid JSON.
        with open(
            OUTPUT_JSON,
            "r",
            encoding="utf-8",
        ) as file:

            loaded_data = json.load(file)

        self.assertIsInstance(
            loaded_data,
            dict,
        )


# ---------------------------------------------------------
# Run tests
# ---------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)