import io

import pytest
from fastapi.testclient import TestClient

from nutricheck_nlp.api.app import create_app


@pytest.fixture()
def client():
    with TestClient(create_app()) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_nutrients_reference(client):
    j = client.get("/v1/nutrients").json()
    assert {n["key"] for n in j["nutrients"]} == {"iron", "vitamin_d", "vitamin_b12", "calcium", "folate"}
    assert any(x["analyte"] == "vitamin_d" and x["deficient_below"] == 20 for x in j["reference_ranges"])


def test_analyze_text_ok(client):
    r = client.post("/v1/analyze/text", json={"text": "Vitamin B12: 150 pg/mL. No iron deficiency."})
    assert r.status_code == 200
    j = r.json()
    st = {n["key"]: n["status"] for n in j["nutrients"]}
    assert st["vitamin_b12"] == "deficient" and st["iron"] == "not_deficient"
    assert j["disclaimer"] and j["version"]


def test_response_schema_complete(client):
    j = client.post("/v1/analyze/text", json={"text": "Hemoglobin: 9 g/dL"}).json()
    for key in ("nutrients", "findings", "summary", "labs", "warnings", "needs_human_review", "suggestions"):
        assert key in j
    assert j["findings"][0]["key"] == "anemia" and j["findings"][0]["status"] == "deficient"


@pytest.mark.parametrize("payload", [
    {}, {"text": ""}, {"text": "   "}, {"text": 123}, {"text": None}, {"text": "x", "sex": "Z"},
    {"text": "x", "ocr_confidence": 1.5}, {"text": "x", "ocr_confidence": -0.1}, {"text": "x" * 60000},
    {"texts": ["a"]}, [],
])
def test_invalid_payloads_422(client, payload):
    assert client.post("/v1/analyze/text", json=payload).status_code == 422


def test_malformed_json_422(client):
    r = client.post("/v1/analyze/text", content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_batch(client):
    r = client.post("/v1/analyze/batch", json={"texts": ["Ferritin: 8 ng/mL", "Vitamin D: 50 ng/mL"]})
    j = r.json()
    assert j["count"] == 2
    assert j["results"][0]["nutrients"][0]["status"] == "deficient"
    assert j["results"][1]["nutrients"][1]["status"] == "not_deficient"


@pytest.mark.parametrize("payload", [{"texts": []}, {"texts": [""]}, {"texts": ["ok", " "]}, {"texts": ["a"] * 51}])
def test_batch_invalid(client, payload):
    assert client.post("/v1/analyze/batch", json=payload).status_code == 422


def test_cors_header(client):
    r = client.options("/v1/analyze/text", headers={"Origin": "https://my-site.com",
                                                    "Access-Control-Request-Method": "POST"})
    assert r.headers.get("access-control-allow-origin") in ("*", "https://my-site.com")


def test_unknown_route_404(client):
    assert client.get("/nope").status_code == 404


def test_wrong_method_405(client):
    assert client.get("/v1/analyze/text").status_code == 405


def test_image_rejects_bad_extension(client):
    r = client.post("/v1/analyze/image", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_image_rejects_empty(client):
    assert client.post("/v1/analyze/image", files={"file": ("a.jpg", b"", "image/jpeg")}).status_code == 400


def _png_bytes() -> bytes:
    """A real (tiny) PNG: uploads are now validated by magic bytes, not just by file name."""
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (20, 10), "white").save(buf, format="PNG")
    return buf.getvalue()


def test_image_too_large(client, monkeypatch):
    import nutricheck_nlp.document_extractor as de
    monkeypatch.setattr(de, "MAX_FILE_BYTES", 10)
    r = client.post("/v1/analyze/image", files={"file": ("a.png", _png_bytes(), "image/png")})
    assert r.status_code == 413


def test_image_not_an_image_is_rejected(client):
    """A file named .jpg that is not a JPEG (renamed/corrupt) must be refused, never sent to the OCR."""
    r = client.post("/v1/analyze/image", files={"file": ("a.jpg", b"not-an-image", "image/jpeg")})
    assert r.status_code == 415


def test_image_without_ocr_gives_503_or_works(client):
    r = client.post("/v1/analyze/image", files={"file": ("a.png", _png_bytes(), "image/png")})
    assert r.status_code in (200, 503)      # 503 when PaddleOCR is not installed


def test_image_pipeline_with_fake_ocr(monkeypatch):
    """Whole /analyze/image path using a stub OCR that returns Person-1-shaped output."""
    app = create_app()

    class FakeOCR:
        def process_image(self, path, preprocess=True):
            return {"success": True, "text": "Ferritin: 7 ng/mL\nVitamin D: 40 ng/mL", "average_confidence": 0.97,
                    "line_count": 2, "image": str(path)}

    with TestClient(app) as c:
        app.state.engine._ocr = FakeOCR()
        r = c.post("/v1/analyze/image", files={"file": ("rep.png", _png_bytes(), "image/png")})
        assert r.status_code == 200
        j = r.json()
        st = {n["key"]: n["status"] for n in j["nutrients"]}
        assert st["iron"] == "deficient" and st["vitamin_d"] == "not_deficient"
        assert j["ocr"]["average_confidence"] == 0.97


def test_api_key_enforced(monkeypatch):
    monkeypatch.setenv("NUTRICHECK_API_KEY", "secret")
    with TestClient(create_app()) as c:
        assert c.post("/v1/analyze/text", json={"text": "x"}).status_code == 401
        assert c.post("/v1/analyze/text", json={"text": "x"}, headers={"X-API-Key": "bad"}).status_code == 401
        ok = c.post("/v1/analyze/text", json={"text": "x"}, headers={"X-API-Key": "secret"})
        assert ok.status_code == 200
        assert c.get("/health").status_code == 200  # health stays public


def test_biogpt_unavailable_degrades_gracefully(client):
    j = client.post("/v1/analyze/text", json={"text": "Ferritin: 8 ng/mL", "use_biogpt": True}).json()
    assert j["biogpt"]["available"] in (True, False)
    assert j["nutrients"][0]["status"] == "deficient"  # rules unaffected


def test_concurrent_requests(client):
    from concurrent.futures import ThreadPoolExecutor
    def call(i):
        return client.post("/v1/analyze/text", json={"text": f"Ferritin: {i % 10 + 1} ng/mL"}).json()["nutrients"][0]["status"]
    with ThreadPoolExecutor(8) as ex:
        assert set(ex.map(call, range(40))) == {"deficient"}


def test_chat_endpoint(client):
    res = client.post("/v1/analyze/text", json={"text": "Vitamin B12: 150 pg/mL"}).json()
    chat_res = client.post("/v1/chat", json={
        "query": "Why is my Vitamin B12 low?",
        "result": res,
        "report_text": "Vitamin B12: 150 pg/mL"
    })
    assert chat_res.status_code == 200
    assert "Vitamin B12" in chat_res.json()["reply"]
