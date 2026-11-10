from pathlib import Path
import sys

sys.path.insert(
    0,
    str(Path("person1-ocr/src").resolve())
)

from ocr_translation_pipeline import process_medical_report


result = process_medical_report(
    image_path="person1-ocr/outputs/sample_report_preprocessed.jpg",
    source_language="pt",
    output_dir="person1-ocr/outputs/pipeline_test"
)

print("\nOCR STATUS: Completed")
print("TRANSLATION STATUS:", result["translation_result"]["status"])

print("\nENGLISH TRANSLATION:")
print(result["translation_result"]["translated_text"])

print("\nSAVED FILES:")
print(result["ocr_json_path"])
print(result["translation_json_path"])