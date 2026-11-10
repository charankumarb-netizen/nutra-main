# Person 3 — NLP engine (English), API and UI

See the top-level `README.md` and `docs/PERSON3_NLP_GUIDE.md`.

```bash
pip install -r ../requirements-dev.txt
python -m pytest -q                                           # 700+ tests
uvicorn nutricheck_nlp.api.app:app --app-dir src --reload     # http://localhost:8000/docs
streamlit run streamlit_app.py
```
Environment variables: `NUTRICHECK_API_KEY`, `NUTRICHECK_CORS_ORIGINS`, `NUTRICHECK_MAX_UPLOAD_MB`, `NUTRICHECK_VAULT_DIR`, `NUTRICHECK_RETENTION_DAYS`,
`NUTRICHECK_DISABLE_CLASSIFIER`, `NUTRICHECK_CLASSIFIER_POLICY`, `NUTRICHECK_CONTEXT_MODEL`, `NUTRICHECK_BIOGPT_MODEL`, `NUTRICHECK_OCR_SRC`.
