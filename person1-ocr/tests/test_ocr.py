"""Manual PaddleOCR smoke script (needs PaddleOCR + model download).

Run it directly:   python tests/test_ocr.py
It used to execute the OCR at *import* time, which made `pytest` crash while collecting the
tests (and its image path only worked from one specific working directory). The logic is
unchanged; it now runs only when executed as a script and the path is file-relative.
"""
import os
from pathlib import Path

# Workaround for PaddlePaddle 3.3.1 CPU + oneDNN issue
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"

IMAGE_PATH = Path(__file__).resolve().parent / "sample_images" / "sample_report.jpg"


def main() -> None:
    from paddleocr import PaddleOCR

    print("Starting PaddleOCR...")
    ocr = PaddleOCR(lang="en", enable_mkldnn=False)

    print("Reading image:", IMAGE_PATH)
    result = ocr.predict(str(IMAGE_PATH))

    print("\n===== OCR RESULT =====")
    for page in result:
        print(page)


if __name__ == "__main__":
    main()
