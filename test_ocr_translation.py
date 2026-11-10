
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path("person1-ocr/src").resolve()))

from ocr_engine import MedicalOCREngine
from translation.pipeline_adapter import translate_ocr_output


def main():
    image_path = Path(
        "person1-ocr/outputs/sample_report_preprocessed.jpg"
    )

    # Step 1: Run OCR
    ocr_engine = MedicalOCREngine()
    ocr_result = ocr_engine.process_image(
        image_path,
        preprocess=False
    )

    # Step 2: Save original OCR output
    Path("person1-ocr/outputs").mkdir(
        parents=True, exist_ok=True
    )

    with open(
        "person1-ocr/outputs/integration_ocr.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(ocr_result, file, indent=4, ensure_ascii=False)

    print("\nOCR extraction completed.")
    print("Extracted text:")
    print(ocr_result.get("text", ""))

    # Step 3: Translate OCR output
    translation_result = translate_ocr_output(
        ocr_result,
        source_language="pt"
    )

    # Step 4: Save translated output
    with open(
        "person1-ocr/outputs/integration_translation.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            translation_result,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nTranslation result:")
    print(json.dumps(
        translation_result,
        indent=4,
        ensure_ascii=False
    ))


if __name__ == "__main__":
    main()
