import sys
from pathlib import Path

OCR_SRC = Path(__file__).resolve().parent / "person1-ocr" / "src"
sys.path.insert(0, str(OCR_SRC))

from ocr_translation_pipeline import process_medical_report


def main():
    image_path = input("Enter medical report image path: ").strip()

    try:
        result = process_medical_report(
            image_path=image_path,
            source_language="pt"
        )

        print("\nOCR completed successfully.")
        print("\nEnglish translation:")
        print(result["translation_result"]["translated_text"])

        print("\nOutput files:")
        print(result["ocr_json_path"])
        print(result["translation_json_path"])

    except FileNotFoundError as error:
        print(f"\nError: {error}")

    except Exception as error:
        print(f"\nSomething went wrong: {error}")


if __name__ == "__main__":
    main()