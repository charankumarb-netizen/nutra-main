
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from ocr_engine import MedicalOCREngine

SAMPLES = {
    "pt": CURRENT_DIR / "sample_images" / "sample_report.jpg",
    "en": CURRENT_DIR / "sample_images" / "sample_report_en.jpg",
}

for language, image_path in SAMPLES.items():
    print(f"\nTesting language: {language}")

    engine = MedicalOCREngine(language=language, cpu_threads=4)
    result = engine.process_image(str(image_path), preprocess=True)

    print("Success:", result["success"])
    print("Language:", result["language"])
    print("Extracted text:")
    print(result["text"])

    if not result["success"] or not result["text"].strip():
        raise RuntimeError(f"OCR failed for language: {language}")

print("\nBoth Portuguese and English OCR tests completed!")
