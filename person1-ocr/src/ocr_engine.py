"""
NutriCheck - Person 1
Medical Document OCR Engine

Responsibilities:
    1. Validate the input medical-record image.
    2. Optionally preprocess the image.
    3. Run PaddleOCR.
    4. Extract recognized text and confidence scores.
    5. Return clean structured data for the NLP module.
    6. Optionally save OCR results as JSON.

Important:
    This module performs OCR only.
    It does NOT diagnose diseases or interpret medical results.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# PaddlePaddle 3.3.1 CPU compatibility settings
# ---------------------------------------------------------------------------
# These must be set BEFORE importing PaddleOCR/PaddlePaddle.
# MKL-DNN/oneDNN caused an inference error in the current environment.
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")

import cv2

try:  # PaddleOCR is heavy/optional: let the module import even when it is not installed
    from paddleocr import PaddleOCR
except ImportError:  # pragma: no cover - depends on the environment
    PaddleOCR = None  # MedicalOCREngine() raises a clear error instead (see __init__)

from image_preprocessor import MedicalImagePreprocessor


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)
# NOTE: logging.basicConfig() used to run here at import time, which overrides the logging
# setup of any application importing this module. It now runs only for manual execution
# (see the __main__ block at the bottom).


# ---------------------------------------------------------------------------
# OCR Engine
# ---------------------------------------------------------------------------

class MedicalOCREngine:
    """
    Reusable OCR engine for medical-record images.

    The engine:
        - validates the input image
        - optionally preprocesses the image
        - performs OCR
        - extracts recognized text
        - calculates confidence
        - returns structured OCR data

    Parameters
    ----------
    language:
        PaddleOCR language code (default "en": NutriCheck interprets English reports only).

    cpu_threads:
        Number of CPU threads used by PaddleOCR.
    """

    SUPPORTED_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    }

    def __init__(
        self,
        language: str = "en",
        cpu_threads: int = 4,
    ) -> None:

        if PaddleOCR is None:
            raise RuntimeError(
                "PaddleOCR is not installed. Run: pip install -r requirements-ocr.txt"
            )

        self.language = language
        self.cpu_threads = cpu_threads

        # Image preprocessing is kept separate from the OCR engine.
        self.preprocessor = MedicalImagePreprocessor()

        logger.info("Initializing PaddleOCR...")

        self.ocr = PaddleOCR(
            lang=self.language,

            # Required for the current PaddlePaddle 3.3.1
            # CPU environment.
            enable_mkldnn=False,

            cpu_threads=self.cpu_threads,
        )

        logger.info("PaddleOCR initialized successfully.")

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    def _validate_image(
        self,
        image_path: str | Path,
    ) -> Path:
        """
        Validate that the input image exists, has a supported extension,
        and can be opened successfully by OpenCV.
        """

        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Medical image not found: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Input path is not a file: {path}"
            )

        extension = path.suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format: {extension}. "
                f"Supported formats: "
                f"{sorted(self.SUPPORTED_EXTENSIONS)}"
            )

        image = cv2.imread(str(path))

        if image is None:
            raise ValueError(
                f"OpenCV could not read the image: {path}"
            )

        height, width = image.shape[:2]

        if width <= 0 or height <= 0:
            raise ValueError(
                f"Invalid image dimensions: {width}x{height}"
            )

        logger.info(
            "Image validated: %s (%dx%d)",
            path.name,
            width,
            height,
        )

        return path

    # -----------------------------------------------------------------------
    # Image preprocessing
    # -----------------------------------------------------------------------

    def _preprocess_image(
        self,
        image_path: Path,
    ) -> Any:
        """
        Preprocess an image using MedicalImagePreprocessor.

        The actual preprocessing implementation remains in
        image_preprocessor.py.

        This method only provides a clean internal interface for the
        OCR engine.
        """

        logger.info(
            "Preprocessing image: %s",
            image_path.name,
        )

        processed_image = self.preprocessor.preprocess(
            image_path
        )

        if processed_image is None:
            raise ValueError(
                "Image preprocessing returned None."
            )

        logger.info(
            "Image preprocessing completed."
        )

        return processed_image

    # -----------------------------------------------------------------------
    # PaddleOCR result conversion
    # -----------------------------------------------------------------------

    @staticmethod
    def _result_to_dict(
        result: Any,
    ) -> dict:
        """
        Convert PaddleOCR's result object into a Python dictionary.

        Supports common PaddleOCR 3.x result representations.
        """

        # PaddleOCR 3.x Result object
        if hasattr(result, "json"):

            data = result.json

            if callable(data):
                data = data()

            if isinstance(data, str):
                try:
                    return json.loads(data)

                except json.JSONDecodeError:
                    return {
                        "raw_result": data
                    }

            if isinstance(data, dict):
                return data

        # Dictionary result
        if isinstance(result, dict):
            return result

        # Fallback
        return {
            "raw_result": str(result)
        }

    # -----------------------------------------------------------------------
    # Extract OCR lines
    # -----------------------------------------------------------------------

    @staticmethod
    def _extract_lines(
        result_data: dict,
    ) -> list[dict]:
        """
        Extract recognized text, confidence scores, and bounding boxes
        from PaddleOCR 3.x results.
        """

        # PaddleOCR 3.x commonly stores OCR data inside "res".
        res = result_data.get(
            "res",
            result_data,
        )

        if not isinstance(res, dict):
            return []

        texts = res.get(
            "rec_texts",
            [],
        )

        scores = res.get(
            "rec_scores",
            [],
        )

        boxes = res.get(
            "rec_boxes",
            [],
        )

        if texts is None:
            texts = []

        if scores is None:
            scores = []

        if boxes is None:
            boxes = []

        lines: list[dict] = []

        for index, text in enumerate(texts):

            text = str(text).strip()

            if not text:
                continue

            # ---------------------------------------------------------------
            # Confidence
            # ---------------------------------------------------------------

            confidence = None

            if index < len(scores):

                try:
                    confidence = float(
                        scores[index]
                    )

                except (
                    TypeError,
                    ValueError,
                ):
                    confidence = None

            # ---------------------------------------------------------------
            # Bounding box
            # ---------------------------------------------------------------

            box = None

            if index < len(boxes):

                try:

                    current_box = boxes[index]

                    if hasattr(
                        current_box,
                        "tolist",
                    ):
                        box = current_box.tolist()

                    else:
                        box = current_box

                except Exception:
                    box = None

            # ---------------------------------------------------------------
            # Structured line
            # ---------------------------------------------------------------

            lines.append(
                {
                    "text": text,
                    "confidence": confidence,
                    "box": box,
                }
            )

        return lines

    # -----------------------------------------------------------------------
    # Calculate average confidence
    # -----------------------------------------------------------------------

    @staticmethod
    def _calculate_average_confidence(
        lines: list[dict],
    ) -> float | None:
        """
        Calculate average confidence from valid OCR lines.
        """

        confidence_values = [
            line["confidence"]
            for line in lines
            if isinstance(
                line.get("confidence"),
                (int, float),
            )
        ]

        if not confidence_values:
            return None

        return sum(
            confidence_values
        ) / len(confidence_values)

    # -----------------------------------------------------------------------
    # Main OCR function
    # -----------------------------------------------------------------------

    def extract_text(
        self,
        image_path: str | Path,
        preprocess: bool = True,
    ) -> dict:
        """
        Run OCR on a medical-record image.

        Parameters
        ----------
        image_path:
            Path to the input medical image.

        preprocess:
            If True, use MedicalImagePreprocessor before OCR.
            If False, send the original image directly to PaddleOCR.

        Returns
        -------
        dict
            Structured OCR result.

        Example
        -------
        {
            "success": True,
            "language": "en",
            "image": "...",
            "preprocessing": True,
            "text": "...",
            "lines": [...],
            "line_count": 7,
            "average_confidence": 0.95,
            "processing_time_seconds": 1.234
        }
        """

        # ---------------------------------------------------------------
        # Validate image BEFORE starting OCR timer.
        # ---------------------------------------------------------------

        path = self._validate_image(
            image_path
        )

        start_time = time.perf_counter()

        logger.info(
            "Running OCR on: %s",
            path,
        )

        try:

            # -----------------------------------------------------------
            # Optional preprocessing
            # -----------------------------------------------------------

            if preprocess:

                processed_image = (
                    self._preprocess_image(path)
                )

                results = self.ocr.predict(
                    processed_image
                )

            else:

                logger.info(
                    "Preprocessing disabled."
                )

                results = self.ocr.predict(
                    str(path)
                )

            # -----------------------------------------------------------
            # Extract lines
            # -----------------------------------------------------------

            all_lines: list[dict] = []

            for result in results:

                result_data = (
                    self._result_to_dict(
                        result
                    )
                )

                lines = (
                    self._extract_lines(
                        result_data
                    )
                )

                all_lines.extend(lines)

            # -----------------------------------------------------------
            # Create clean text
            # -----------------------------------------------------------

            clean_text = "\n".join(
                line["text"]
                for line in all_lines
                if line.get("text")
            )

            # -----------------------------------------------------------
            # Average confidence
            # -----------------------------------------------------------

            average_confidence = (
                self._calculate_average_confidence(
                    all_lines
                )
            )

            # -----------------------------------------------------------
            # Processing time
            # -----------------------------------------------------------

            processing_time = (
                time.perf_counter()
                - start_time
            )

            # -----------------------------------------------------------
            # Final structured result
            # -----------------------------------------------------------

            output = {
                "success": True,
                "language": self.language,
                "image": str(path),
                "preprocessing": preprocess,
                "text": clean_text,
                "lines": all_lines,
                "line_count": len(all_lines),
                "average_confidence": (
                    average_confidence
                ),
                "processing_time_seconds": round(
                    processing_time,
                    3,
                ),
            }

            logger.info(
                "OCR completed: %d lines detected",
                len(all_lines),
            )

            return output

        except Exception as exc:

            processing_time = (
                time.perf_counter()
                - start_time
            )

            logger.exception(
                "OCR processing failed."
            )

            return {
                "success": False,
                "language": self.language,
                "image": str(path),
                "preprocessing": preprocess,
                "text": "",
                "lines": [],
                "line_count": 0,
                "average_confidence": None,
                "processing_time_seconds": round(
                    processing_time,
                    3,
                ),
                "error": str(exc),
            }

    # -----------------------------------------------------------------------
    # Convenience method
    # -----------------------------------------------------------------------

    def process_image(
        self,
        image_path: str | Path,
        preprocess: bool = True,
    ) -> dict:
        """
        Convenience wrapper around extract_text().

        This gives the rest of the NutriCheck application a simple
        entry point:

            engine.process_image(image_path)

        The actual OCR implementation remains inside extract_text().
        """

        return self.extract_text(
            image_path=image_path,
            preprocess=preprocess,
        )

    # -----------------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------------

    @staticmethod
    def save_json(
        result: dict,
        output_path: str | Path,
    ) -> Path:
        """
        Save OCR output as UTF-8 JSON.
        """

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                result,
                file,
                ensure_ascii=False,
                indent=4,
            )

        logger.info(
            "OCR result saved to: %s",
            output_path,
        )

        return output_path


# ---------------------------------------------------------------------------
# Local test / manual execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # -----------------------------------------------------------------------
    # Project paths
    # -----------------------------------------------------------------------

    PROJECT_ROOT = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    SAMPLE_IMAGE = (
        PROJECT_ROOT
        / "tests"
        / "sample_images"
        / "sample_report.jpg"
    )

    OUTPUT_DIR = (
        PROJECT_ROOT
        / "outputs"
    )

    OUTPUT_FILE = (
        OUTPUT_DIR
        / "sample_report_ocr.json"
    )

    # -----------------------------------------------------------------------
    # Header
    # -----------------------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "NutriCheck - Medical OCR Engine"
    )

    print(
        "=" * 60
    )

    # -----------------------------------------------------------------------
    # Initialize engine
    # -----------------------------------------------------------------------

    engine = MedicalOCREngine(
        language="en",
        cpu_threads=4,
    )

    # -----------------------------------------------------------------------
    # Run OCR
    # -----------------------------------------------------------------------

    result = engine.process_image(
        SAMPLE_IMAGE,
        preprocess=True,
    )

    # -----------------------------------------------------------------------
    # Display result
    # -----------------------------------------------------------------------

    if result["success"]:

        print(
            "\n===== EXTRACTED TEXT ====="
        )

        print(
            result["text"]
        )

        print(
            "\n===== OCR INFORMATION ====="
        )

        print(
            f"Lines detected: "
            f"{result['line_count']}"
        )

        print(
            f"Average confidence: "
            f"{result['average_confidence']}"
        )

        print(
            f"Preprocessing: "
            f"{result['preprocessing']}"
        )

        print(
            f"Processing time: "
            f"{result['processing_time_seconds']} "
            f"seconds"
        )

        # ---------------------------------------------------------------
        # Save JSON
        # ---------------------------------------------------------------

        engine.save_json(
            result,
            OUTPUT_FILE,
        )

        print(
            f"\nJSON saved to:\n{OUTPUT_FILE}"
        )

    else:

        print(
            "\nOCR FAILED"
        )

        print(
            result.get(
                "error",
                "Unknown OCR error.",
            )
        )