"""
NutriCheck - Person 1
Medical Document Image Preprocessor

Uses OpenCV to prepare medical-record photographs for OCR.

Responsibilities:
    1. Load and validate the input image.
    2. Resize oversized images when necessary.
    3. Improve local contrast.
    4. Reduce image noise.
    5. Sharpen text regions.
    6. Return a processed image for OCR.
    7. Optionally save the processed image.

Important:
    - The original image is NEVER modified.
    - Preprocessing is optional.
    - This module does NOT perform OCR.
    - This module does NOT interpret medical information.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Medical Image Preprocessor
# ---------------------------------------------------------------------------

class MedicalImagePreprocessor:
    """
    OpenCV-based preprocessing pipeline for medical document images.

    The preprocessing pipeline is:

        Load
          ↓
        Resize if necessary
          ↓
        Contrast enhancement
          ↓
        Noise reduction
          ↓
        Sharpening
          ↓
        Return processed image

    All operations are performed on an in-memory copy.
    The source image on disk is never modified.
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
        target_width: int = 1600,
        enable_resize: bool = True,
        enable_contrast: bool = True,
        enable_denoising: bool = True,
        enable_sharpening: bool = True,
    ) -> None:
        """
        Initialize the preprocessing pipeline.

        Args:
            target_width:
                Maximum width used when resizing large images.

            enable_resize:
                Enable/disable resizing.

            enable_contrast:
                Enable/disable CLAHE contrast enhancement.

            enable_denoising:
                Enable/disable noise reduction.

            enable_sharpening:
                Enable/disable text sharpening.
        """

        if target_width <= 0:
            raise ValueError(
                "target_width must be greater than 0."
            )

        self.target_width = target_width

        self.enable_resize = enable_resize
        self.enable_contrast = enable_contrast
        self.enable_denoising = enable_denoising
        self.enable_sharpening = enable_sharpening

    # -----------------------------------------------------------------------
    # Validate image path
    # -----------------------------------------------------------------------

    def _validate_path(
        self,
        image_path: str | Path,
    ) -> Path:
        """
        Validate the image path before loading.
        """

        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Image not found: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Input path is not a file: {path}"
            )

        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format: {path.suffix}. "
                f"Supported formats: "
                f"{sorted(self.SUPPORTED_EXTENSIONS)}"
            )

        return path

    # -----------------------------------------------------------------------
    # Load image
    # -----------------------------------------------------------------------

    def load_image(
        self,
        image_path: str | Path,
    ) -> np.ndarray:
        """
        Load an image using OpenCV.

        Returns:
            BGR OpenCV image as a NumPy array.
        """

        path = self._validate_path(
            image_path
        )

        image = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise ValueError(
                f"OpenCV could not read image: {path}"
            )

        if image.size == 0:
            raise ValueError(
                f"Loaded image is empty: {path}"
            )

        height, width = image.shape[:2]

        if height <= 0 or width <= 0:
            raise ValueError(
                f"Invalid image dimensions: "
                f"{width}x{height}"
            )

        logger.info(
            "Loaded image: %s (%dx%d)",
            path.name,
            width,
            height,
        )

        return image

    # -----------------------------------------------------------------------
    # Resize image
    # -----------------------------------------------------------------------

    def resize_image(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Resize oversized images.

        Images smaller than target_width are not enlarged.
        """

        if image is None or image.size == 0:
            raise ValueError(
                "Cannot resize an empty image."
            )

        height, width = image.shape[:2]

        # Do not resize if resizing is disabled.
        if not self.enable_resize:
            return image

        # Do not enlarge smaller images.
        if width <= self.target_width:
            return image

        scale = self.target_width / width

        new_width = self.target_width
        new_height = max(
            1,
            int(height * scale),
        )

        resized = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )

        logger.debug(
            "Image resized: %dx%d -> %dx%d",
            width,
            height,
            new_width,
            new_height,
        )

        return resized

    # -----------------------------------------------------------------------
    # Enhance contrast
    # -----------------------------------------------------------------------

    def enhance_contrast(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Improve local contrast using CLAHE.

        LAB color space is used so that contrast enhancement
        primarily affects luminance rather than color.
        """

        if image is None or image.size == 0:
            raise ValueError(
                "Cannot enhance contrast on an empty image."
            )

        if not self.enable_contrast:
            return image

        lab = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2LAB,
        )

        l_channel, a_channel, b_channel = cv2.split(
            lab
        )

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )

        enhanced_l = clahe.apply(
            l_channel
        )

        enhanced_lab = cv2.merge(
            (
                enhanced_l,
                a_channel,
                b_channel,
            )
        )

        enhanced = cv2.cvtColor(
            enhanced_lab,
            cv2.COLOR_LAB2BGR,
        )

        return enhanced

    # -----------------------------------------------------------------------
    # Reduce noise
    # -----------------------------------------------------------------------

    def reduce_noise(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Reduce photographic noise while attempting to preserve text.
        """

        if image is None or image.size == 0:
            raise ValueError(
                "Cannot reduce noise on an empty image."
            )

        if not self.enable_denoising:
            return image

        denoised = cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h=5,
            hColor=5,
            templateWindowSize=7,
            searchWindowSize=21,
        )

        return denoised

    # -----------------------------------------------------------------------
    # Sharpen image
    # -----------------------------------------------------------------------

    def sharpen(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Apply moderate sharpening to improve text edges.

        A moderate kernel is intentionally used so that
        characters are not excessively sharpened.
        """

        if image is None or image.size == 0:
            raise ValueError(
                "Cannot sharpen an empty image."
            )

        if not self.enable_sharpening:
            return image

        kernel = np.array(
            [
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0],
            ],
            dtype=np.float32,
        )

        sharpened = cv2.filter2D(
            image,
            -1,
            kernel,
        )

        return sharpened

    # -----------------------------------------------------------------------
    # Complete processing pipeline
    # -----------------------------------------------------------------------

    def process_image(
        self,
        image_path: str | Path,
    ) -> np.ndarray:
        """
        Run the complete preprocessing pipeline.

        This is the preferred public method.

        Args:
            image_path:
                Path to the original image.

        Returns:
            Processed OpenCV BGR image.

        Important:
            The original image file is never modified.
        """

        logger.info(
            "Starting image preprocessing: %s",
            image_path,
        )

        image = self.load_image(
            image_path
        )

        # Work with an independent in-memory copy.
        processed = image.copy()

        if self.enable_resize:
            processed = self.resize_image(
                processed
            )

        if self.enable_contrast:
            processed = self.enhance_contrast(
                processed
            )

        if self.enable_denoising:
            processed = self.reduce_noise(
                processed
            )

        if self.enable_sharpening:
            processed = self.sharpen(
                processed
            )

        if processed is None or processed.size == 0:
            raise ValueError(
                "Preprocessing produced an empty image."
            )

        logger.info(
            "Image preprocessing completed."
        )

        return processed

    # -----------------------------------------------------------------------
    # Backward-compatible method
    # -----------------------------------------------------------------------

    def preprocess(
        self,
        image_path: str | Path,
    ) -> np.ndarray:
        """
        Backward-compatible alias for process_image().

        Your current ocr_engine.py already calls:

            self.preprocessor.preprocess(path)

        so you do not need to change that code immediately.
        """

        return self.process_image(
            image_path
        )

    # -----------------------------------------------------------------------
    # Save processed image
    # -----------------------------------------------------------------------

    def save_processed_image(
        self,
        image: np.ndarray,
        output_path: str | Path,
    ) -> Path:
        """
        Save a processed image to disk.

        The destination directory is created automatically.
        """

        if image is None or image.size == 0:
            raise ValueError(
                "Cannot save an empty image."
            )

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        success = cv2.imwrite(
            str(output_path),
            image,
        )

        if not success:
            raise IOError(
                f"Could not save processed image: "
                f"{output_path}"
            )

        logger.info(
            "Processed image saved: %s",
            output_path,
        )

        return output_path


# ---------------------------------------------------------------------------
# Local test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    PROJECT_ROOT = (
        Path(__file__).resolve().parents[1]
    )

    SAMPLE_IMAGE = (
        PROJECT_ROOT
        / "tests"
        / "sample_images"
        / "sample_report.jpg"
    )

    OUTPUT_IMAGE = (
        PROJECT_ROOT
        / "outputs"
        / "sample_report_preprocessed.jpg"
    )

    print("\n" + "=" * 60)
    print("NutriCheck - Image Preprocessor")
    print("=" * 60)

    try:

        preprocessor = MedicalImagePreprocessor(
            target_width=1600,
            enable_resize=True,
            enable_contrast=True,
            enable_denoising=True,
            enable_sharpening=True,
        )

        processed_image = (
            preprocessor.process_image(
                SAMPLE_IMAGE
            )
        )

        preprocessor.save_processed_image(
            processed_image,
            OUTPUT_IMAGE,
        )

        print(
            "\nPreprocessing completed successfully."
        )

        print(
            f"Output:\n{OUTPUT_IMAGE}"
        )

    except Exception as exc:

        logger.exception(
            "Preprocessing failed."
        )

        print(
            f"\nPreprocessing failed: {exc}"
        )