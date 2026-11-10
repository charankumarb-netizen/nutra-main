"""The Hugging Face Space folders must work in isolation (copied package only)."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_spaces import build  # noqa: E402


@pytest.fixture(scope="module")
def spaces(tmp_path_factory):
    return build(tmp_path_factory.mktemp("dist"))


def test_expected_files(spaces):
    backend, ui = spaces
    for f in ("Dockerfile", "README.md", "requirements.txt", "src/nutricheck_nlp/api/app.py"):
        assert (backend / f).exists(), f
    for f in ("streamlit_app.py", "README.md", "requirements.txt", "src/nutricheck_nlp/pipeline.py"):
        assert (ui / f).exists(), f
    assert "sdk: docker" in (backend / "README.md").read_text() and "app_port: 7860" in (backend / "README.md").read_text()
    assert "sdk: streamlit" in (ui / "README.md").read_text()
    assert "7860" in (backend / "Dockerfile").read_text()


def test_backend_runs_isolated(spaces):
    backend, _ = spaces
    code = (
        "from fastapi.testclient import TestClient\n"
        "from nutricheck_nlp.api.app import app\n"
        "c=TestClient(app)\n"
        "r=c.post('/v1/analyze/text',json={'text':'Ferritin: 8 ng/mL'})\n"
        "assert r.status_code==200 and r.json()['nutrients'][0]['status']=='deficient'\n"
        "print('OK')\n"
    )
    out = subprocess.run([sys.executable, "-W", "ignore", "-c", code], cwd=backend, capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": str(backend / "src")})
    assert "OK" in out.stdout, out.stderr[-800:]


def test_streamlit_space_runs_isolated(spaces):
    _, ui = spaces
    code = (
        "from streamlit.testing.v1 import AppTest\n"
        "at=AppTest.from_file('streamlit_app.py',default_timeout=60).run()\n"
        "assert not at.exception\n"
        "at.text_area[0].set_value('Vitamin D: 10 ng/mL').run(); at.button[0].click().run()\n"
        "assert not at.exception\n"
        "assert any('Deficiency detected' in m.value for m in at.metric)\n"
        "print('OK')\n"
    )
    out = subprocess.run([sys.executable, "-W", "ignore", "-c", code], cwd=ui, capture_output=True, text=True,
                         env=os.environ.copy())
    assert "OK" in out.stdout, out.stderr[-800:]


def test_package_has_no_heavy_imports_at_startup():
    code = "import sys; sys.path.insert(0,'src'); import nutricheck_nlp; print('torch' in sys.modules, 'transformers' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert out.stdout.strip() == "False False"


def test_spaces_ship_the_trained_model_and_parser_dependencies(spaces):
    backend, ui = spaces
    for d in (backend, ui):
        assert (d / "models" / "context_clf.joblib").is_file()
        req = (d / "requirements.txt").read_text()
        assert all(x in req for x in ("pymupdf", "python-docx", "scikit-learn", "joblib", "python-multipart"))
    assert "COPY --chown=user models ./models" in (backend / "Dockerfile").read_text()


def test_backend_isolated_upload_and_classifier(spaces):
    """The copied package + models folder alone must serve a PDF upload and report the trained classifier."""
    backend, _ = spaces
    code = (
        "import fitz\n"
        "from fastapi.testclient import TestClient\n"
        "from nutricheck_nlp.api.app import app\n"
        "doc=fitz.open(); doc.new_page().insert_text((50,60),'Ferritin: 8 ng/mL'); pdf=doc.tobytes()\n"
        "with TestClient(app) as c:\n"
        "    r=c.post('/v1/analyze/file',files={'file':('a.pdf',pdf,'application/pdf')})\n"
        "    assert r.status_code==200, r.text\n"
        "    assert r.json()['results'][0]['nutrients'][0]['status']=='deficient'\n"
        "    assert c.get('/v1/models/status').json()['context_classifier']['available'] is True\n"
        "print('OK')\n"
    )
    out = subprocess.run([sys.executable, "-W", "ignore", "-c", code], cwd=backend, capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": str(backend / "src")})
    assert "OK" in out.stdout, out.stderr[-1200:]


def test_frontend_example_is_safe_and_complete():
    html = (ROOT / "frontend_example" / "index.html").read_text(encoding="utf-8")
    assert "innerHTML" not in html                       # report text is user data: it must go through textContent (no XSS)
    for needle in ("/v1/analyze/text", "/v1/analyze/file", "<progress", "type=\"file\"", "Download JSON", "Download report CSV", "evidence_items"):
        assert needle in html, needle
