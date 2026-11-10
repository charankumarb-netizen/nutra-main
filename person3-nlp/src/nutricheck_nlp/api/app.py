"""
NutriCheck FastAPI service.

Run locally:   uvicorn nutricheck_nlp.api.app:app --app-dir src --reload --port 8000
Docs:          http://localhost:8000/docs

Endpoints
    GET  /health                health check
    GET  /v1/models/status      which components are REALLY available (rules, trained classifier, BioGPT, OCR, parsers)
    GET  /v1/nutrients          nutrients, analytes and the reference table used
    POST /v1/analyze/text       typed clinical text (JSON)
    POST /v1/analyze/file       PDF / DOCX / TXT / CSV / image upload (multipart)
    POST /v1/analyze/batch      many texts at once (JSON)
    POST /v1/analyze/image      kept for compatibility, same as /v1/analyze/file for images
    POST /v1/records/analyze    several reports: per-report analysis + health timeline + contradiction / missing-info check
    POST /v1/records/analyze/files   same, with uploaded files
    POST /v1/explain            explanation at a chosen level (simple / detailed / medical terms / Telugu / Hindi)
    POST /v1/visit-prep         one-page doctor-visit summary (Markdown + printable HTML)
    POST /v1/privacy/mask       mask names / phones / IDs in a research copy;  GET /v1/privacy/notice
    POST|GET|DELETE /v1/vault/...   private health vault (consent required; header X-Vault-Key)

Security notes: uploads are validated (extension, MIME type, magic bytes, size) and processed in memory; nothing is stored,
request bodies are never logged. Set NUTRICHECK_API_KEY to require an X-API-Key header and NUTRICHECK_CORS_ORIGINS to your
real frontend domain before any public deployment (never expose the key in browser JavaScript).
"""

from __future__ import annotations

import hmac
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from .. import __version__
from ..context_model import load_default_classifier
from .. import document_extractor as de
from ..document_extractor import ALLOWED_EXT, DocumentError
from ..lexicon import ANALYTES, NUTRIENT_KEYS, TARGETS, reference_table
from ..pipeline import NutriCheckNLP
from ..chatbot import NutriCheckChatbot
from ..crosscheck import cross_check
from ..explain import explain as explain_report
from ..privacy import mask_pii
from ..timeline import build_timeline
from ..vault import HealthVault, VaultError
from ..visit_prep import build_visit_prep
from .schemas import (
    AnalysisResponse, AnalyzeRequest, BatchRequest, BatchResponse, ChatRequest, ExplainRequest, FileAnalysisResponse, MaskRequest,
    RecordsRequest, VaultStoreRequest, VisitPrepRequest,
)

logger = logging.getLogger("nutricheck.api")

ALLOWED_FILE_EXT = sorted(ALLOWED_EXT)


def _origins() -> list[str]:
    raw = os.getenv("NUTRICHECK_CORS_ORIGINS", "*")
    return [o.strip() for o in raw.split(",") if o.strip()]


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Optional protection: set NUTRICHECK_API_KEY to require the X-API-Key header."""
    expected = os.getenv("NUTRICHECK_API_KEY")
    if expected and not hmac.compare_digest((x_api_key or "").encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _build_engine(classifier: Any) -> NutriCheckNLP:
    if classifier == "auto":
        classifier = None if os.getenv("NUTRICHECK_DISABLE_CLASSIFIER") == "1" else load_default_classifier()
    return NutriCheckNLP(
        classifier=classifier,
        classifier_policy=os.getenv("NUTRICHECK_CLASSIFIER_POLICY", "conflict"),
        classifier_threshold=float(os.getenv("NUTRICHECK_CLASSIFIER_THRESHOLD", "0.85")),
    )


def create_app(classifier: Any = "auto", engine: NutriCheckNLP | None = None, vault_dir: str | None = None) -> FastAPI:
    """
    classifier: "auto" (load the shipped/NUTRICHECK_CONTEXT_MODEL model if present), None (rules only), or any
                object with predict(sentences) -> [(label, probability)].
    engine:     inject a ready NutriCheckNLP (tests, custom OCR).
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = engine or _build_engine(classifier)  # created once, reused by every request
        if _origins() == ["*"] and not os.getenv("NUTRICHECK_API_KEY"):
            logger.warning("CORS is open (*) and no API key is set: restrict NUTRICHECK_CORS_ORIGINS before public deployment.")
        yield

    app = FastAPI(
        title="NutriCheck NLP API",
        version=__version__,
        description="Nutrient-deficiency screening support from ENGLISH medical records "
                    "(rule-based clinical NLP + trained context classifier + optional BioGPT second opinion), with evidence, "
                    "timeline, contradiction checks, plain-language explanations and a privacy-first vault.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=_origins(), allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"],
    )

    def get_engine() -> NutriCheckNLP:
        eng = getattr(app.state, "engine", None)
        if eng is None:  # e.g. TestClient used without the lifespan context
            eng = app.state.engine = engine or _build_engine(classifier)
        return eng

    @app.get("/health", tags=["system"])
    def health():
        return {"status": "ok", "version": __version__}

    @app.get("/v1/models/status", tags=["system"], dependencies=[Depends(require_api_key)])
    def models_status():
        return get_engine().models_status()

    @app.get("/v1/nutrients", tags=["reference"], dependencies=[Depends(require_api_key)])
    def nutrients():
        return {
            "nutrients": [
                {"key": k, "name": TARGETS[k].name} for k in NUTRIENT_KEYS
            ],
            "reference_ranges": reference_table(),
            "analytes": sorted(ANALYTES),
        }

    @app.post("/v1/analyze/text", response_model=AnalysisResponse, tags=["analysis"],
              dependencies=[Depends(require_api_key)])
    def analyze_text(req: AnalyzeRequest):
        return get_engine().analyze_text(
            req.text, sex=req.sex, ocr_confidence=req.ocr_confidence,
            use_biogpt=req.use_biogpt, explain=req.explain, use_classifier=req.use_classifier,
        )

    @app.post("/v1/analyze/batch", response_model=BatchResponse, tags=["analysis"],
              dependencies=[Depends(require_api_key)])
    def analyze_batch(req: BatchRequest):
        eng = get_engine()
        results = [eng.analyze_text(t, use_biogpt=req.use_biogpt, use_classifier=req.use_classifier) for t in req.texts]
        return {"count": len(results), "results": results}

    async def _read_upload(file: UploadFile, request: Request) -> bytes:
        limit = de.MAX_FILE_BYTES                                  # read at call time (configurable / testable)
        too_big = HTTPException(413, f"File too large (maximum {limit / (1024 * 1024):g} MB).")
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > limit + 64 * 1024:   # + multipart overhead
            raise too_big
        data = await file.read(limit + 1)                          # never read more than the limit into memory
        if len(data) > limit:
            raise too_big
        return data

    @app.post("/v1/analyze/file", response_model=FileAnalysisResponse, tags=["analysis"],
              dependencies=[Depends(require_api_key)])
    async def analyze_file(
        request: Request,
        file: UploadFile = File(..., description="PDF, DOCX, TXT, CSV or image (jpg/png/bmp/tif/webp)"),
        sex: str | None = Form(None, description="M or F (optional)"),
        use_biogpt: bool = Form(False),
        use_classifier: bool | None = Form(None),
    ):
        if sex not in (None, "", "M", "F"):
            raise HTTPException(422, "sex must be 'M' or 'F'")
        data = await _read_upload(file, request)
        try:
            results = await run_in_threadpool(
                get_engine().analyze_file, file.filename or "", data, file.content_type,
                sex=sex or None, use_biogpt=use_biogpt, use_classifier=use_classifier,
            )
        except DocumentError as exc:
            raise HTTPException(exc.status, exc.message)
        finally:
            del data                                            # nothing is kept after the request
        return {"document": results[0]["document"], "count": len(results), "results": results}

    @app.post("/v1/analyze/image", response_model=AnalysisResponse, tags=["analysis"],
              dependencies=[Depends(require_api_key)])
    async def analyze_image(request: Request, file: UploadFile = File(...)):
        """Compatibility endpoint: images only (OCR + NLP). Prefer /v1/analyze/file."""
        from pathlib import Path

        if Path(file.filename or "").suffix.lower() in (".pdf", ".docx", ".txt", ".csv"):
            raise HTTPException(415, "This endpoint accepts images only; use /v1/analyze/file.")
        data = await _read_upload(file, request)
        try:
            results = await run_in_threadpool(get_engine().analyze_file, file.filename or "", data, file.content_type)
        except DocumentError as exc:
            raise HTTPException(exc.status, exc.message)
        return results[0]


    # ------------------------------------------------------------------ several records: timeline + contradictions
    def _records_response(results: list[dict]) -> dict:
        tl = build_timeline(results)
        cc = cross_check(results)
        return {
            "reports": [{"report_id": r["report_id"], "needs_human_review": r["needs_human_review"], "quality": r["quality"],
                         "summary": r["summary"], "warnings": r["warnings"]} for r in results],
            "timeline": tl, "crosscheck": cc,
            "_results": results,
        }

    @app.post("/v1/records/analyze", tags=["records"], dependencies=[Depends(require_api_key)])
    def records_analyze(req: RecordsRequest):
        eng = get_engine()
        results = []
        for i, rep in enumerate(req.reports):
            r = eng.analyze_text(rep.text, sex=rep.sex, date_order=req.date_order)
            r["report_id"] = rep.label or f"report-{i + 1}"
            results.append(r)
        out = _records_response(results)
        out["analyses"] = out.pop("_results")
        return out

    @app.post("/v1/records/analyze/files", tags=["records"], dependencies=[Depends(require_api_key)])
    async def records_analyze_files(request: Request, files: list[UploadFile] = File(...),
                                    date_order: str | None = Form(None)):
        if date_order not in (None, "", "DMY", "MDY"):
            raise HTTPException(422, "date_order must be DMY or MDY")
        if not 1 <= len(files) <= 10:
            raise HTTPException(422, "Upload between 1 and 10 files.")
        results: list[dict] = []
        for f in files:
            data = await _read_upload(f, request)
            try:
                recs = await run_in_threadpool(get_engine().analyze_file, f.filename or "", data, f.content_type, date_order=date_order or None)
            except DocumentError as exc:
                raise HTTPException(exc.status, f"{f.filename}: {exc.message}")
            for r in recs:
                r["report_id"] = (f.filename or "file") + (f" [{r['record_id']}]" if len(recs) > 1 else "")
                results.append(r)
        out = _records_response(results)
        out["analyses"] = out.pop("_results")
        return out

    # ------------------------------------------------------------------ explanations / visit preparation / privacy
    @app.post("/v1/explain", tags=["explain"], dependencies=[Depends(require_api_key)])
    def explain_endpoint(req: ExplainRequest):
        res = get_engine().analyze_text(req.text, sex=req.sex)
        return {"analysis_summary": res["summary"], "quality": res["quality"], "explanation": explain_report(res, req.level)}

    @app.post("/v1/visit-prep", tags=["explain"], dependencies=[Depends(require_api_key)])
    def visit_prep(req: VisitPrepRequest):
        eng = get_engine()
        results = []
        for i, rep in enumerate(req.reports):
            r = eng.analyze_text(rep.text, sex=rep.sex, date_order=req.date_order)
            r["report_id"] = rep.label or f"report-{i + 1}"
            results.append(r)
        return build_visit_prep(results, build_timeline(results), cross_check(results), req.language, req.generated_on)

    @app.post("/v1/chat", tags=["chat"], dependencies=[Depends(require_api_key)])
    def chat_endpoint(req: ChatRequest):
        bot = NutriCheckChatbot(result=req.result, report_text=req.report_text)
        return {"reply": bot.answer(req.query)}

    @app.post("/v1/privacy/mask", tags=["privacy"], dependencies=[Depends(require_api_key)])
    def privacy_mask(req: MaskRequest):
        return mask_pii(req.text)

    @app.get("/v1/privacy/notice", tags=["privacy"])
    def privacy_notice():
        return HealthVault.notice(get_vault().retention_days)

    # ------------------------------------------------------------------ private vault
    def get_vault() -> HealthVault:
        v = getattr(app.state, "vault", None)
        if v is None:
            v = app.state.vault = HealthVault(vault_dir or os.getenv("NUTRICHECK_VAULT_DIR", "vault_data"))
        return v

    def vault_call(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except VaultError as exc:
            raise HTTPException(exc.status, str(exc))

    @app.post("/v1/vault/reports", tags=["vault"], dependencies=[Depends(require_api_key)])
    def vault_store(req: VaultStoreRequest, x_vault_key: str = Header(default="")):
        if req.consent is not True:
            raise HTTPException(403, "Storing a report requires explicit permission (consent=true). Nothing was stored.")
        res = get_engine().analyze_text(req.text, sex=req.sex, date_order=req.date_order)
        return vault_call(get_vault().store, x_vault_key, res, consent=req.consent, label=req.label)

    @app.get("/v1/vault/reports", tags=["vault"], dependencies=[Depends(require_api_key)])
    def vault_list(x_vault_key: str = Header(default="")):
        return {"reports": vault_call(get_vault().list, x_vault_key)}

    @app.get("/v1/vault/reports/{report_id}", tags=["vault"], dependencies=[Depends(require_api_key)])
    def vault_get(report_id: str, x_vault_key: str = Header(default="")):
        return vault_call(get_vault().get, x_vault_key, report_id)

    @app.delete("/v1/vault/reports/{report_id}", tags=["vault"], dependencies=[Depends(require_api_key)])
    def vault_delete(report_id: str, x_vault_key: str = Header(default="")):
        return vault_call(get_vault().delete, x_vault_key, report_id)

    @app.delete("/v1/vault/reports", tags=["vault"], dependencies=[Depends(require_api_key)])
    def vault_delete_all(x_vault_key: str = Header(default="")):
        return vault_call(get_vault().delete_all, x_vault_key)

    @app.get("/v1/vault/timeline", tags=["vault"], dependencies=[Depends(require_api_key)])
    def vault_timeline(x_vault_key: str = Header(default="")):
        return vault_call(get_vault().timeline, x_vault_key)

    @app.get("/v1/vault/export", tags=["vault"], dependencies=[Depends(require_api_key)])
    def vault_export(x_vault_key: str = Header(default="")):
        return vault_call(get_vault().export, x_vault_key)

    return app


app = create_app()
