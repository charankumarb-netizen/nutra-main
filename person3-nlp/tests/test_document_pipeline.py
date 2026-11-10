"""End-to-end: bytes of a PDF / DOCX / CSV / TXT / image  ->  extraction -> NLP -> evidence with page and line."""
import warnings

import pytest

warnings.filterwarnings("ignore", category=FutureWarning)

from doc_factory import FakeOCR, make_docx, make_pdf, make_png_text, make_scanned_pdf  # noqa: E402
from nutricheck_nlp import NutriCheckNLP  # noqa: E402
from nutricheck_nlp.document_extractor import DocumentError  # noqa: E402


@pytest.fixture()
def eng():
    return NutriCheckNLP(ocr_engine=FakeOCR())


def st(res):
    return {n["key"]: n["status"] for n in res["nutrients"] + res["findings"]}


def item(res, key, typ):
    n = next(n for n in res["nutrients"] + res["findings"] if n["key"] == key)
    return [i for i in n["evidence_items"] if i["type"] == typ]


def test_pdf_evidence_carries_page_and_line(eng):
    pdf = make_pdf(["Patient: Ana Souza\nSex: Female\nVitamin D: 18 ng/mL", "Notes\nFerritin: 8 ng/mL"])
    (res,) = eng.analyze_file("laudo.pdf", pdf, "application/pdf")
    assert st(res)["vitamin_d"] == "deficient" and st(res)["iron"] == "deficient"
    vd, fe = item(res, "vitamin_d", "lab")[0], item(res, "iron", "lab")[0]
    assert vd["location"]["page"] == 1 and vd["location"]["line"] == 3
    assert fe["location"]["page"] == 2 and fe["location"]["line"] == 2
    assert res["document"]["kind"] == "pdf" and res["document"]["page_count"] == 2 and res["patient"]["sex"] == "F"


def test_pdf_table_with_lab_reference(eng):
    pdf = make_pdf(["Results"], tables={0: [["Test", "Result", "Unit", "Reference"],
                                              ["Ferritin", "8", "ng/mL", "15 - 150"], ["Vitamin D", "45", "ng/mL", "30 - 100"]]})
    (res,) = eng.analyze_file("t.pdf", pdf)
    assert st(res)["iron"] == "deficient" and st(res)["vitamin_d"] == "not_deficient"
    assert item(res, "iron", "lab")[0]["lab_reference"]["low"] == 15 and not res["needs_human_review"]
    assert res["document"]["tables"][0]["rows"] == 3


def test_docx_with_text_and_table(eng):
    docx = make_docx(["No vitamin B12 deficiency.", "Notes: vegetarian patient."],
                     [[["Test", "Result", "Unit"], ["Ferritin", "6", "ng/mL"], ["Calcium", "9.5", "mg/dL"]]])
    (res,) = eng.analyze_file("a.docx", docx)
    assert st(res) | {} == st(res) and st(res)["vitamin_b12"] == "not_deficient"
    assert st(res)["iron"] == "deficient" and st(res)["calcium"] == "not_deficient"
    assert item(res, "iron", "lab")[0]["location"]["page"] is None             # Word has no fixed pages


def test_csv_many_patients_one_result_each(eng):
    csv = b"patient,Ferritin (ng/mL),Vitamin D (ng/mL),Sex\nA,8,50,F\nB,90,12,M\nC,70,40,F\n"
    results = eng.analyze_file("batch.csv", csv, "text/csv")
    assert [r["record_id"] for r in results] == ["A", "B", "C"]
    assert [st(r)["iron"] for r in results] == ["deficient", "not_deficient", "not_deficient"]
    assert [st(r)["vitamin_d"] for r in results] == ["not_deficient", "deficient", "not_deficient"]
    assert results[1]["patient"]["sex"] == "M"


def test_csv_text_notes(eng):
    csv = "id;text\n1;Patient with iron deficiency.\n2;No iron deficiency.\n3;Suspected B12 deficiency.\n".encode()
    r = eng.analyze_file("n.csv", csv)
    assert [st(x)["iron"] for x in r[:2]] == ["deficient", "not_deficient"] and st(r[2])["vitamin_b12"] == "suspected"
    assert r[2]["needs_human_review"]


def test_txt_file(eng):
    (res,) = eng.analyze_file("a.txt", "Hemoglobin: 9 g/dL\nIron deficiency anemia.".encode("cp1252"))
    assert st(res)["anemia"] == "deficient" and st(res)["iron"] == "deficient"


def test_image_through_fake_ocr(eng):
    (res,) = eng.analyze_file("foto.png", make_png_text(), "image/png")
    assert st(res)["iron"] == "deficient" and st(res)["vitamin_d"] == "not_deficient"
    assert res["ocr"]["average_confidence"] == 0.97 and not res["needs_human_review"]


def test_image_low_ocr_confidence_forces_review():
    (res,) = NutriCheckNLP(ocr_engine=FakeOCR(confidence=0.6)).analyze_file("foto.png", make_png_text())
    assert res["needs_human_review"] and any("OCR" in w for w in res["warnings"])

def test_scanned_pdf_with_and_without_ocr():
    (res,) = NutriCheckNLP(
        ocr_engine=FakeOCR("Ferritin: 7 ng/mL")
    ).analyze_file("s.pdf", make_scanned_pdf())

    assert st(res)["iron"] == "deficient"
    assert item(res, "iron", "lab")[0]["location"]["page"] == 1

    # Simulate OCR being unavailable, even when PaddleOCR is installed.
    class UnavailableOCR:
        def process_image(self, path):
            raise RuntimeError("OCR engine unavailable (simulated test failure)")

    (res2,) = NutriCheckNLP(
        ocr_engine=UnavailableOCR()
    ).analyze_file("s.pdf", make_scanned_pdf())

    assert res2["needs_human_review"]
    assert st(res2)["iron"] == "not_assessed"
    assert any("OCR" in w for w in res2["warnings"])



def test_empty_document_asks_for_review(eng):
    (res,) = eng.analyze_file("v.pdf", make_pdf([""]))
    assert res["needs_human_review"] and set(st(res).values()) == {"not_assessed"}


def test_invalid_files_raise_document_error(eng):
    for name, data in [("x.exe", b"MZ"), ("x.pdf", b"nope"), ("x.docx", b"nope"), ("x.csv", b"")]:
        with pytest.raises(DocumentError):
            eng.analyze_file(name, data)


def test_evidence_items_for_text_mentions(eng):
    (res,) = eng.analyze_file("n.txt", b"Line 1\nThe patient denies iron deficiency.\nFerritin low.")
    items = item(res, "iron", "text")
    assert {i["assertion"] for i in items} == {"negated", "affirmed"}
    neg = next(i for i in items if i["assertion"] == "negated")
    assert neg["location"]["line"] == 2 and "denies" in neg["sentence"].lower()


def test_analyze_text_without_pages_still_reports_lines(engine):
    res = engine.analyze_text("Line 1\nVitamin D: 18 ng/mL")
    loc = res["labs"][0]["location"]
    assert loc["page"] is None and loc["line"] == 2


# ------------------------------------------------------------------ Person 1 integration: the OCR path bug
def test_person1_ocr_folder_is_found_and_paddle_missing_is_reported_clearly():
    """
    Regression test for the path bug (parents[3].parent pointed OUTSIDE the project, so Person 1's code was never found).
    With the fix, `ocr_engine` imports from ../person1-ocr/src and fails only because PaddleOCR itself is not installed.
    """
    import importlib.util
    import sys

    eng = NutriCheckNLP()
    if importlib.util.find_spec("paddleocr") is not None:
        pytest.skip("PaddleOCR installed: the real engine would load (heavy); path fix covered by the other branch")
    with pytest.raises(RuntimeError) as e:
        eng._get_ocr()
    assert "PaddleOCR is not installed" in str(e.value) and "No module named 'ocr_engine'" not in str(e.value)
    assert any(p.replace("\\", "/").endswith("person1-ocr/src") for p in sys.path)
    assert "unavailable" in eng.models_status()["ocr"]["error"]
    with pytest.raises(RuntimeError):          # the failure is remembered, not retried on every request
        eng._get_ocr()
