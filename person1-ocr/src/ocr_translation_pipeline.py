
import json
from pathlib import Path
import sys

# Locate the OCR engine
OCR_SRC = Path(__file__).resolve().parent
if str(OCR_SRC) not in sys.path:
    sys.path.insert(0, str(OCR_SRC))

from ocr_engine import MedicalOCREngine
from translation.pipeline_adapter import translate_ocr_output


def process_medical_report(
    image_path,
    source_language="pt",
    output_dir=None
):
    """
    Extract text from a medical report image,
    translate it into English, and save both results.
    """

    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Medical report image not found: {image_path}"
        )

    if output_dir is None:
        output_dir = image_path.parent

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: OCR extraction
    ocr_engine = MedicalOCREngine()
    ocr_result = ocr_engine.process_image(
        image_path,
        preprocess=True
    )

    # Step 2: Save original OCR JSON
    ocr_json_path = output_dir / "ocr_output.json"

    with open(ocr_json_path, "w", encoding="utf-8") as file:
        json.dump(
            ocr_result,
            file,
            indent=4,
            ensure_ascii=False
        )

    # Step 3: Translate OCR output
    translation_result = translate_ocr_output(
        ocr_result,
        source_language=source_language
    )

    # Step 4: Save translated JSON
    translation_json_path = (
        output_dir / "translated_output.json"
    )

    with open(
        translation_json_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            translation_result,
            file,
            indent=4,
            ensure_ascii=False
        )

    return {
        "ocr_result": ocr_result,
        "translation_result": translation_result,
        "ocr_json_path": str(ocr_json_path),
        "translation_json_path": str(translation_json_path)
    }
