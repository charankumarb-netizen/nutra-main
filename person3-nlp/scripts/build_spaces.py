"""
Assemble two ready-to-push Hugging Face Space folders:

    dist/hf_backend    (Docker Space  -> FastAPI)
    dist/hf_streamlit  (Streamlit Space -> UI, local engine or remote API)

Usage:  python scripts/build_spaces.py [--out dist]
Then:   cd dist/hf_backend && git init && git remote add space https://huggingface.co/spaces/<user>/<name> && git add . && git commit -m init && git push space main
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TPL = ROOT / "deploy" / "templates"


def _copy_pkg(dst: Path) -> None:
    shutil.copytree(ROOT / "src" / "nutricheck_nlp", dst / "src" / "nutricheck_nlp",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def _copy_models(dst: Path) -> None:
    """The trained context classifier lives in <app>/models (context_model.DEFAULT_MODEL_PATH)."""
    src = ROOT / "models"
    if src.is_dir():
        shutil.copytree(src, dst / "models", ignore=shutil.ignore_patterns("__pycache__", "_tiny_tokenizer", "*.tmp"))
    else:
        (dst / "models").mkdir(parents=True)          # Dockerfile COPY needs the folder; the API then runs rules-only


def build(out: Path) -> tuple[Path, Path]:
    backend, ui = out / "hf_backend", out / "hf_streamlit"
    for d in (backend, ui):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    _copy_pkg(backend)
    _copy_models(backend)
    shutil.copy(TPL / "backend_Dockerfile", backend / "Dockerfile")
    shutil.copy(TPL / "backend_README.md", backend / "README.md")
    shutil.copy(TPL / "backend_requirements.txt", backend / "requirements.txt")
    _copy_pkg(ui)
    _copy_models(ui)
    shutil.copy(ROOT / "streamlit_app.py", ui / "streamlit_app.py")
    shutil.copy(TPL / "streamlit_README.md", ui / "README.md")
    shutil.copy(TPL / "streamlit_requirements.txt", ui / "requirements.txt")
    return backend, ui


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "dist"))
    a = ap.parse_args()
    b, u = build(Path(a.out))
    print(f"Backend Space : {b}\nStreamlit Space: {u}")
