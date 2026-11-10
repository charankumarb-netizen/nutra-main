# NutriCheck — Nutrient Deficiency Screening Support from English Medical Records (Project 17)

| Part | Owner | Folder |
|---|---|---|
| OCR (PaddleOCR + preprocessing) | Person 1 | `person1-ocr/` |
| Dataset validation / statistics | Person 2 | `person2-dataset/` |
| **NLP engine · document parsing · trained classifier · BioGPT · FastAPI · Streamlit · tests · HF deploy** | **Person 3** | `person3-nlp/` |

Flow: `PDF / DOCX / CSV / TXT / image → text extraction (OCR for scans, Person 1) → language guard → English NLP engine → evidence-based decision + quality checks → JSON → Streamlit / website`

**Language scope.** The engine reads **English** clinical text only (normalisation, tokenisation, negation, units, number formats, terminology). Text in any other
language is *detected and refused* (`not_assessed`, "please provide the report in English") instead of being guessed at. There is no translation step. Telugu and Hindi appear only
as **output templates** for explanations (see feature 3) — the system does not read Telugu or Hindi reports.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python run_all_tests.py                                   # tests of all three parts + one summary
cd person3-nlp
uvicorn nutricheck_nlp.api.app:app --app-dir src --reload # API  -> http://localhost:8000/docs
streamlit run streamlit_app.py                            # UI   -> http://localhost:8501
python scripts/train_context_classifier.py                # (re)train the context classifier
python scripts/evaluate.py --synthetic 5000               # accuracy report on gold + synthetic cases
python scripts/build_spaces.py                            # dist/hf_backend + dist/hf_streamlit for Hugging Face
```
Optional: `requirements-ocr.txt` (PaddleOCR: images and scanned PDFs) · `requirements-biogpt.txt` (BioGPT, fine-tuning an English biomedical BERT).

## The eight added features (all in `person3-nlp/src/nutricheck_nlp/`)
| # | Feature | Module | API |
|---|---|---|---|
| 1 | **Medical Report Detective**: exact sentence, value, unit, date, page/section/line, context, missing information | `evidence.py` | `detective` in every nutrient of `/v1/analyze/*` |
| 2 | **Health Timeline Intelligence**: dates + comparable units → trends flagged for review (not a diagnosis) | `dates.py`, `timeline.py` | `POST /v1/records/analyze` |
| 3 | **Explain My Report at My Level**: simple / detailed English, medical terms with glossary, Telugu, Hindi; every sentence cites evidence and is re-verified | `explain.py`, `i18n.py` | `POST /v1/explain` |
| 4 | **Contradiction and Missing-Information Detector** across records | `crosscheck.py` | `POST /v1/records/analyze` |
| 5 | **AI that knows when not to predict**: 5 quality checks, `reliability`, abstention as a safe outcome | `quality.py` | `quality` in every response |
| 6 | **One-click document understanding**: format, type, language, per-page processing report, values to verify | `document_extractor.py`, `pipeline.py` | `processing_report` in `/v1/analyze/file` |
| 7 | **Doctor-Visit Preparation**: one-page Markdown + printable HTML, questions from the actual findings | `visit_prep.py` | `POST /v1/visit-prep` |
| 8 | **Privacy-First Health Vault**: consent, masked summaries, separate originals, delete, export, retention | `privacy.py`, `vault.py` | `/v1/vault/*`, `/v1/privacy/*` |

**Read `docs/PERSON3_NLP_GUIDE.md`** — pipeline, features, safety rules, measured accuracy and the honest limitations (including why "100 % correct prediction" cannot be promised).

## What changed in Person 1's and Person 2's files
* `person1-ocr/src/ocr_engine.py`: default OCR language is now `"en"` (was `"pt"`); `PaddleOCR` import guarded; `logging.basicConfig()` no longer runs at import.
* `person1-ocr/tests/`: heavy code no longer runs at import; new `test_ocr_logic_offline.py`; added `sample_images/sample_report_en.jpg` (generated English sample; no OCR result is claimed for it).
* `person2-dataset/src/*`: `find_dataset()` no longer hard-coded to Colab. The MTSamples data is English, which the engine now reads directly.

> Decision-support software only — not a medical device, not a diagnosis. Passing the included tests is not clinical validation.
