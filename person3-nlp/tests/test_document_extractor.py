"""extract_pdf / extract_docx / extract_csv / extract_txt / extract_image and upload validation, on REAL generated files."""
import warnings

import pytest

warnings.filterwarnings("ignore", category=FutureWarning)

from doc_factory import FakeOCR, make_docx, make_encrypted_pdf, make_pdf, make_png_text, make_scanned_pdf  # noqa: E402
from nutricheck_nlp import document_extractor as de  # noqa: E402
from nutricheck_nlp.document_extractor import (  # noqa: E402
    DocumentError, extract_csv, extract_docx, extract_document, extract_image, extract_pdf, extract_txt,
    table_to_lines, validate_upload,
)

LAB_ROWS = [["Test", "Result", "Unit", "Reference"], ["Ferritin", "8", "ng/mL", "15 - 150"],
            ["Vitamin B12", "150", "pg/mL", "200 - 900"]]


# ------------------------------------------------------------------ TXT
@pytest.mark.parametrize("enc", ["utf-8", "utf-8-sig", "cp1252", "latin-1"])
def test_txt_encodings(enc):
    d = extract_txt("Iron deficiency. Calcium: 7.8 mg/dL".encode(enc))
    assert "Iron deficiency" in d.text and "Calcium" in d.text


def test_txt_empty_warns():
    assert "empty" in " ".join(extract_txt(b"   \n ").warnings)


def test_txt_crlf_and_blank_lines_normalised():
    assert extract_txt(b"a\r\n\r\n\r\n\r\nb").text == "a\n\nb"


# ------------------------------------------------------------------ PDF
def test_pdf_digital_text_and_pages():
    d = extract_pdf(make_pdf(["Vitamin D: 18 ng/mL", "Ferritin: 8 ng/mL"]))
    assert d.page_count == 2 and not d.needs_review
    spans = d.records[0].pages
    assert [s["page"] for s in spans] == [1, 2] and spans[0]["end"] <= spans[1]["start"]
    assert "Vitamin D: 18 ng/mL" in d.text and "Ferritin: 8 ng/mL" in d.text


def test_pdf_table_becomes_analyte_lines():
    d = extract_pdf(make_pdf(["Report"], tables={0: LAB_ROWS}))
    assert "Ferritin: 8 ng/mL (Ref: 15 - 150)" in d.text
    assert "Vitamin B12: 150 pg/mL (Ref: 200 - 900)" in d.text      # spaces inside cells preserved
    assert d.tables and d.tables[0]["rows"] == 3 and d.records[0].pages[0]["method"] == "table"


def test_pdf_scanned_without_ocr_is_flagged():
    d = extract_pdf(make_scanned_pdf())
    assert d.needs_review and not d.text.strip()
    assert any("OCR is not available" in w for w in d.warnings)


def test_pdf_scanned_with_ocr_callback():
    calls = []

    def ocr(png):
        calls.append(len(png))
        return {"text": "Ferritin: 8 ng/mL", "average_confidence": 0.93}

    d = extract_pdf(make_scanned_pdf(2), ocr_page=ocr)
    assert len(calls) == 2 and png_ok(calls) and d.ocr_confidence == pytest.approx(0.93)
    assert d.records[0].pages[0]["method"] == "ocr" and "Ferritin: 8 ng/mL" in d.text and not d.needs_review


def png_ok(sizes):
    return all(s > 100 for s in sizes)


def test_pdf_ocr_failure_is_reported_not_raised():
    def boom(png):
        raise RuntimeError("engine crashed")

    d = extract_pdf(make_scanned_pdf(), ocr_page=boom)
    assert d.needs_review and any("OCR failed" in w for w in d.warnings)


def test_pdf_mixed_digital_and_scanned_pages():
    import fitz

    doc = fitz.open()
    doc.new_page().insert_text((50, 60), "Vitamin D: 18 ng/mL")
    doc.new_page().insert_image(fitz.Rect(40, 40, 400, 200), stream=make_png_text("x"))
    d = extract_pdf(doc.tobytes(), ocr_page=lambda png: {"text": "Ferritin: 8 ng/mL", "average_confidence": 0.9})
    assert [s["method"] for s in d.records[0].pages] == ["text", "ocr"]


def test_pdf_encrypted_corrupt_and_empty():
    with pytest.raises(DocumentError, match="password") as e:
        extract_pdf(make_encrypted_pdf())
    assert e.value.status == 422
    with pytest.raises(DocumentError):
        extract_pdf(b"%PDF-1.4 this is not really a pdf")


def test_pdf_page_limit(monkeypatch):
    monkeypatch.setattr(de, "MAX_PDF_PAGES", 2)
    d = extract_pdf(make_pdf(["a: 1", "b: 2", "c: 3", "d: 4"]))
    assert d.page_count == 4 and len(d.records[0].pages) == 2 and any("only the first 2" in w for w in d.warnings)


def test_pdf_blank_page_needs_review():
    d = extract_pdf(make_pdf([""]))
    assert d.needs_review and "No text" in " ".join(d.warnings)


# ------------------------------------------------------------------ DOCX
def test_docx_paragraphs_and_table_in_order():
    d = extract_docx(make_docx(["Report", "No iron deficiency."], [LAB_ROWS]))
    lines = d.text.split("\n")
    assert lines[:2] == ["Report", "No iron deficiency."]
    assert "Ferritin: 8 ng/mL (Ref: 15 - 150)" in lines and d.tables[0]["lines"] == 2


def test_docx_title_row_above_header_and_merged_cells():
    d = extract_docx(make_docx([], [[["Test results", "", ""], ["Test", "Result", "Unit"], ["Ferritin", "8", "ng/mL"]]],
                               merge_first_row=True))
    assert d.text == "Ferritin: 8 ng/mL"


def test_docx_corrupt_and_empty():
    with pytest.raises(DocumentError):
        extract_docx(b"PK\x03\x04 broken")
    assert extract_docx(make_docx([""])).needs_review


# ------------------------------------------------------------------ CSV
def test_csv_long_format_lab_table():
    d = extract_csv("Test,Result,Unit,Reference\nFerritin,8,ng/mL,15 - 150\nVitamin D,45,ng/mL,30 - 100\n".encode())
    assert len(d.records) == 1
    assert d.text == "Ferritin: 8 ng/mL (Ref: 15 - 150)\nVitamin D: 45 ng/mL (Ref: 30 - 100)"


def test_csv_wide_format_one_record_per_row():
    d = extract_csv("patient,Ferritin (ng/mL),Vitamin B12 (pg/mL),Sex\nA,8,150,F\nB,90,500,M\n".encode())
    assert [r.record_id for r in d.records] == ["A", "B"]
    assert "Ferritin (ng/mL): 8" in d.records[0].text and "Sex: M" in d.records[1].text


def test_csv_free_text_column_semicolon_separator():
    d = extract_csv("id;text\n1;Patient with iron deficiency.\n2;No B12 deficiency.\n".encode())
    assert [(r.record_id, r.text) for r in d.records] == [("1", "Patient with iron deficiency."), ("2", "No B12 deficiency.")]


def test_csv_long_format_grouped_by_patient():
    d = extract_csv("patient,test,result,unit\nA,Ferritin,8,ng/mL\nA,Vitamin D,50,ng/mL\nB,Ferritin,90,ng/mL\n".encode())
    assert [r.record_id for r in d.records] == ["A", "B"] and d.records[0].text.count("\n") == 1


def test_csv_encoding_bom_and_latin1():
    assert "Ferritin: 8 ng/mL" in extract_csv("\ufeffTest,Result,Unit\nFerritin,8,ng/mL\n".encode("utf-8")).text
    assert "Calcium: 7.8 mg/dL" in extract_csv("Test;Result;Unit\nCalcium;7.8;mg/dL\n".encode("latin-1")).text


def test_csv_row_limit(monkeypatch):
    monkeypatch.setattr(de, "MAX_CSV_ROWS", 3)
    d = extract_csv(("id,text\n" + "\n".join(f"{i},note {i}" for i in range(10))).encode())
    assert len(d.records) == 3 and any("more than 3 rows" in w for w in d.warnings)


@pytest.mark.parametrize("raw", [b"", b"a,b\n", b"\n\n"])
def test_csv_without_data_rejected(raw):
    with pytest.raises(DocumentError):
        extract_csv(raw)


# ------------------------------------------------------------------ tables
def test_table_to_lines_variants():
    assert table_to_lines(LAB_ROWS)[0] == "Ferritin: 8 ng/mL (Ref: 15 - 150)"
    assert table_to_lines([["Vitamin D", "18 ng/mL"], ["Ferritin", "8 ng/mL"]]) == ["Vitamin D 18 ng/mL", "Ferritin 8 ng/mL"]
    assert table_to_lines([["Ferritin (ng/mL)", "Sex"], ["8", "F"]]) == ["Ferritin (ng/mL): 8", "Sex: F"]
    assert table_to_lines([["", ""], ["", ""]]) == []
    assert table_to_lines([["Test", "Result"], ["Ferritin", ""]]) == []        # a row without a value is dropped


# ------------------------------------------------------------------ images
def test_image_through_ocr_engine(tmp_path):
    f = tmp_path / "a.png"
    f.write_bytes(make_png_text())
    d = extract_image(f, FakeOCR("Ferritin: 7 ng/mL", 0.97))
    assert d.text == "Ferritin: 7 ng/mL" and d.ocr_confidence == 0.97 and not d.needs_review


def test_image_low_confidence_needs_review(tmp_path):
    f = tmp_path / "a.png"
    f.write_bytes(make_png_text())
    assert extract_image(f, FakeOCR(confidence=0.6)).needs_review


def test_image_ocr_unavailable_and_failure(tmp_path):
    f = tmp_path / "a.png"
    f.write_bytes(make_png_text())
    with pytest.raises(DocumentError) as e:
        extract_image(f, None)
    assert e.value.status == 503
    with pytest.raises(DocumentError, match="boom"):
        extract_image(f, FakeOCR(success=False))
    assert extract_image(f, FakeOCR(text="  ")).needs_review


# ------------------------------------------------------------------ validation
@pytest.mark.parametrize("name,data,ctype,status", [
    ("a.exe", b"MZ", None, 415),
    ("noext", b"abc", None, 415),
    ("a.pdf", b"", "application/pdf", 400),
    ("a.pdf", b"not a pdf at all", "application/pdf", 415),            # magic bytes
    ("a.pdf", b"%PDF-1.4", "image/png", 415),                          # MIME vs extension
    ("a.docx", b"PK\x03\x04junk", None, 415),                         # a zip that is not a Word file
    ("a.docx", b"hello", None, 415),
    ("a.png", b"\xff\xd8\xff", "image/png", 415),                      # JPEG bytes in a .png
    ("a.txt", b"abc\x00def", "text/plain", 415),                       # NUL bytes -> binary
    ("a.csv", b"\x00\x01\x02", "text/csv", 415),
    ("a.jpg", b"GIF89a", None, 415),
])
def test_validate_upload_rejections(name, data, ctype, status):
    with pytest.raises(DocumentError) as e:
        validate_upload(name, data, ctype)
    assert e.value.status == status


def test_validate_upload_size_limit(monkeypatch):
    monkeypatch.setattr(de, "MAX_FILE_BYTES", 100)
    with pytest.raises(DocumentError) as e:
        validate_upload("a.txt", b"x" * 101, "text/plain")
    assert e.value.status == 413


@pytest.mark.parametrize("name,data,ctype", [
    ("a.pdf", make_pdf(["x"]), "application/pdf"),
    ("a.PDF", make_pdf(["x"]), None),
    ("a.docx", make_docx(["x"]), "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    ("a.csv", b"a,b\n1.2", "application/vnd.ms-excel"),
    ("a.csv", b"a,b\n1.2", "text/csv; charset=utf-8"),
    ("a.txt", b"hello", "application/octet-stream"),
    ("a.png", make_png_text(), "image/png"),
])
def test_validate_upload_accepts(name, data, ctype):
    assert validate_upload(name, data, ctype) == "." + name.rsplit(".", 1)[1].lower()


def test_dispatcher_routes_every_format(tmp_path):
    ocr = FakeOCR()
    kinds = {
        "a.pdf": make_pdf(["Ferritin: 8 ng/mL"]), "a.docx": make_docx(["Ferritin: 8 ng/mL"]),
        "a.csv": b"Test,Result,Unit\nFerritin,8,ng/mL\n", "a.txt": b"Ferritin: 8 ng/mL", "a.png": make_png_text(),
    }
    out = {name: extract_document(name, data, ocr_engine=ocr).kind for name, data in kinds.items()}
    assert out == {"a.pdf": "pdf", "a.docx": "docx", "a.csv": "csv", "a.txt": "txt", "a.png": "image"} and ocr.calls == 1


def test_image_temp_file_is_removed(tmp_path):
    import glob
    import tempfile

    before = set(glob.glob(tempfile.gettempdir() + "/tmp*.png"))
    extract_document("a.png", make_png_text(), ocr_engine=FakeOCR())
    assert set(glob.glob(tempfile.gettempdir() + "/tmp*.png")) == before
