"""POST /v1/analyze/file, /v1/models/status and the batch endpoint over HTTP."""
import warnings

import pytest
from fastapi.testclient import TestClient

warnings.filterwarnings("ignore", category=FutureWarning)

from doc_factory import FakeOCR, make_docx, make_encrypted_pdf, make_pdf, make_png_text, make_scanned_pdf  # noqa: E402
from nutricheck_nlp import document_extractor as de  # noqa: E402
from nutricheck_nlp.api.app import create_app  # noqa: E402

PDF, DOCX = "application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


@pytest.fixture()
def client():
    app = create_app(classifier=None)
    with TestClient(app) as c:
        app.state.engine._ocr = FakeOCR()
        yield c


def post(client, name, data, ctype, **form):
    return client.post("/v1/analyze/file", files={"file": (name, data, ctype)}, data=form)


def statuses(j, i=0):
    return {n["key"]: n["status"] for n in j["results"][i]["nutrients"]}


def test_pdf_upload(client):
    r = post(client, "laudo.pdf", make_pdf(["Vitamin B12: 150 pg/mL", "Ferritin: 90 ng/mL"]), PDF)
    assert r.status_code == 200
    j = r.json()
    assert j["count"] == 1 and j["document"]["kind"] == "pdf" and j["document"]["page_count"] == 2
    assert statuses(j)["vitamin_b12"] == "deficient" and statuses(j)["iron"] == "not_deficient"
    ev = next(n for n in j["results"][0]["nutrients"] if n["key"] == "vitamin_b12")["evidence_items"][0]
    assert ev["location"]["page"] == 1 and ev["value"] == 150 and ev["unit"] == "pg/ml"


def test_docx_upload(client):
    r = post(client, "a.docx", make_docx(["Denies iron deficiency."], [[["Test", "Result", "Unit"], ["Vitamin D", "12", "ng/mL"]]]), DOCX)
    assert r.status_code == 200 and statuses(r.json())["vitamin_d"] == "deficient" and statuses(r.json())["iron"] == "not_deficient"


def test_csv_upload_with_many_records(client):
    csv = b"patient,Ferritin (ng/mL)\nA,8\nB,90\n"
    j = post(client, "l.csv", csv, "text/csv").json()
    assert j["count"] == 2 and [r["record_id"] for r in j["results"]] == ["A", "B"]
    assert statuses(j, 0)["iron"] == "deficient" and statuses(j, 1)["iron"] == "not_deficient"


def test_txt_upload_and_sex_form_field(client):
    r = post(client, "a.txt", b"Hemoglobin: 12.8 g/dL", "text/plain", sex="M")
    assert r.status_code == 200 and r.json()["results"][0]["findings"][0]["status"] == "deficient"
    r = post(client, "a.txt", b"Hemoglobin: 12.8 g/dL", "text/plain", sex="F")
    assert r.json()["results"][0]["findings"][0]["status"] == "not_deficient"


def test_image_upload_uses_ocr(client):
    r = post(client, "f.png", make_png_text(), "image/png")
    assert r.status_code == 200 and statuses(r.json())["iron"] == "deficient"
    assert r.json()["results"][0]["ocr"]["average_confidence"] == 0.97


def test_scanned_pdf_goes_through_ocr(client):
    r = post(client, "s.pdf", make_scanned_pdf(), PDF)
    assert r.status_code == 200 and statuses(r.json())["iron"] == "deficient"


@pytest.mark.parametrize("name,data,ctype,code", [
    ("a.exe", b"MZ\x90", "application/octet-stream", 415),
    ("a.pdf", b"not a pdf", PDF, 415),
    ("a.pdf", make_pdf(["x"]), "image/png", 415),
    ("a.docx", b"PK\x03\x04bad", DOCX, 415),
    ("a.pdf", b"", PDF, 400),
    ("a.pdf", make_encrypted_pdf(), PDF, 422),
    ("a.csv", b"a,b\n", "text/csv", 422),
    ("noext", b"hello", "text/plain", 415),
])
def test_rejections(client, name, data, ctype, code):
    r = post(client, name, data, ctype)
    assert r.status_code == code and "detail" in r.json()


def test_upload_too_large(client, monkeypatch):
    monkeypatch.setattr(de, "MAX_FILE_BYTES", 200)
    assert post(client, "a.txt", b"x" * 1000, "text/plain").status_code == 413


def test_missing_file_field_422(client):
    assert client.post("/v1/analyze/file").status_code == 422


def test_invalid_sex_422(client):
    assert post(client, "a.txt", b"x", "text/plain", sex="Z").status_code == 422


def test_ocr_missing_gives_503_for_images():
    app = create_app(classifier=None)
    with TestClient(app) as c:
        r = post(c, "f.png", make_png_text(), "image/png")
        assert r.status_code in (503, 200)        # 503 when PaddleOCR is not installed (never a 500)


def test_api_key_protects_file_and_status_endpoints(monkeypatch):
    monkeypatch.setenv("NUTRICHECK_API_KEY", "s3cret")
    with TestClient(create_app(classifier=None)) as c:
        assert post(c, "a.txt", b"x", "text/plain").status_code == 401
        assert c.get("/v1/models/status").status_code == 401
        ok = c.post("/v1/analyze/file", files={"file": ("a.txt", b"Ferritin: 8 ng/mL", "text/plain")}, headers={"X-API-Key": "s3cret"})
        assert ok.status_code == 200 and c.get("/health").status_code == 200


def test_models_status_reports_reality(client):
    j = client.get("/v1/models/status").json()
    assert j["rule_engine"]["available"] and j["context_classifier"]["available"] is False
    assert set(j) >= {"rule_engine", "context_classifier", "biogpt", "ocr", "documents"}
    assert j["documents"]["pymupdf"] and j["documents"]["python_docx"]
    assert j["biogpt"]["loaded"] is False


def test_models_status_with_classifier_loaded():
    with TestClient(create_app()) as c:               # "auto": loads the shipped model
        j = c.get("/v1/models/status").json()
    assert j["context_classifier"]["available"] and "affirmed" in j["context_classifier"]["labels"]
    assert j["context_classifier"]["metrics"]["handwritten_blind"]["accuracy"] > 0.8


def test_classifier_fields_in_text_response_and_toggle():
    with TestClient(create_app()) as c:
        j = c.post("/v1/analyze/text", json={"text": "No iron deficiency."}).json()
        assert j["classifier"]["available"] and j["classifier"]["items"]
        off = c.post("/v1/analyze/text", json={"text": "No iron deficiency.", "use_classifier": False}).json()
        assert off["classifier"] is None


def test_batch_of_texts_with_documents_independent(client):
    j = client.post("/v1/analyze/batch", json={"texts": ["Ferritin: 8 ng/mL", "No iron deficiency."]}).json()
    assert [n["nutrients"][0]["status"] for n in j["results"]] == ["deficient", "not_deficient"]


def test_nothing_is_persisted_after_upload(client, tmp_path, monkeypatch):
    import os
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    post(client, "f.png", make_png_text(), "image/png")
    post(client, "a.pdf", make_pdf(["Ferritin: 8 ng/mL"]), PDF)
    assert os.listdir(tmp_path) == []
