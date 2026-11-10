# Person 3 guide — English NLP engine, evidence, safety features, API

The system turns **English** clinical text and documents into **evidence-based, reviewable findings** about iron, vitamin D, vitamin B12, calcium and folate
(plus anemia as a supporting finding). It flags possible deficiencies for human review. It does not diagnose and does not recommend supplements.

---------------------------------------------------------------------------------------------------------------------------

## 0. Read this first: what "100 % correct" can and cannot mean

No clinical NLP system can honestly promise that every prediction is 100 % correct. Real reports contain ambiguity, OCR damage, missing units, and statements that
contradict the laboratory values. What this project does instead:

* **It refuses to guess.** A missing unit, an OCR-ambiguous digit ("l8"), an implausible value, an invalid number, an unsupported language, or low OCR confidence produces
  `not_assessed` / `suspected` plus `reliability = "abstain"` and an explicit "please verify" action. Abstaining is treated as a *successful, safe* outcome.
* **It never silently resolves a contradiction.** Text vs. laboratory disagreement, or several different values without dates, give `conflicting` + `needs_human_review`.
* **Every result is traceable** to the exact sentence/value/page in the report (feature 1).

What was measured (and what it does **not** prove):

| Check | Result | What it means |
|---|---|---|
| 177 + 193 hand-written gold cases (`tests/gold_cases*.py`) | all pass | the engine behaves as designed; I wrote both the cases and the engine, so this is **not** independent validation |
| 24 hardening cases H01–H24 (`tests/test_hardening_h01_h24.py`) | all pass | context, units, OCR, injection, markup, language, size/time limits |
| 20,000 freshly generated synthetic cases (new seed) | 0 failures | proves consistency with the generator's own grammar only |
| 29 unseen, differently worded realistic notes written after the engine was frozen | 28 correct; the 1 "miss" was a wrong expectation of mine (a value without a unit is correctly only `suspected`) | a small, honest spot check — not a validation study |
| Total test run | `python run_all_tests.py` → ALL GREEN (≈ 750 tests in Person 3) | regression safety |

**To claim real accuracy** you need an independent, clinician-labelled set of real, de-identified English reports (e.g. the English MTSamples-style notes in Person 2's
dataset, labelled by clinicians), measured with sensitivity / specificity / precision / recall on a test set nobody tuned on.

---------------------------------------------------------------------------------------------------------------------------

## 1. Language scope (what was changed from the Portuguese build)

* All language-specific components were rewritten for English: lexicon (aliases, symptoms, risk factors, treatments), negation / uncertainty / history / family / hypothetical cues,
  laboratory-line grammar and reference-range keys, **number format** (decimal point, `1,200` = one thousand two hundred; `12,5` is *flagged as ambiguous*, not read as 12.5),
  date parsing (ISO, month names, numeric dates with an explicit day/month rule), tables and CSV column names, all user-facing messages, the trained classifier's training data,
  the hand-written evaluation sets, Streamlit and the web example. Person 1's OCR default language is now `"en"`.
* **Removed** Portuguese-only assumptions (e.g. decimal-comma parsing, "VR" reference keys, Portuguese stop words, Brazilian lab table headers).
* **Kept** language-independent parts: offset-preserving normalisation, unit conversion and plausibility ranges, the decision engine, document extraction (PDF/DOCX/CSV/TXT/image),
  OCR integration, the classifier architecture, the API structure.
* **No translation step.** Text that is not English is detected by `language_guard.py` (function-word lists + script detection) and **refused**; the system does not claim to
  support any other language. "Unknown" language on a short lab-only line (`Ferritin: 8 ng/mL`) is accepted because the English analyte names match.
* **Test specification adaptation.** The H01–H24 list was written in Portuguese. Each case was ported to its English equivalent with the same logic. **H23** was adapted on purpose:
  the original expected "English is not validated by the Portuguese pipeline". Now English is the supported language, and **Portuguese/other-language text** gets the
  "not validated, no analysis" refusal.
* Telugu/Hindi: **output templates only** (feature 3). The wording was written by me and has **not** been reviewed by a clinician or a professional medical translator; the
  response says so. Numbers, units, dates and quotes are never translated.

---------------------------------------------------------------------------------------------------------------------------

## 2. Pipeline

```
INPUT  text · PDF · DOCX · CSV · TXT · image      document_extractor.py (+ Person 1 OCR for scans/images)
  ▼
language_guard.py   English? (else refuse)          text_normalizer.py   lower-case, ASCII-fold, OCR damage fixed, offsets kept
  ▼
lab_extractor.py    "Ferritin: 8 ng/mL" → value, unit, conversion, reference range, date, temporal cue, safety guards
text_extractor.py   "No iron deficiency" → negation / suspicion / family / history / hypothetical
dates.py            dates + "currently / previously" cues       lexicon.py   analytes, aliases, units, cut-offs (edit thresholds here)
  ▼
context_model.py    trained sentence classifier = VERIFIER (never the decider)
  ▼
pipeline.py         decision engine → statuses + reliability + review reasons
evidence.py         Medical Report Detective     quality.py   "knows when not to predict"
  ▼
timeline.py · crosscheck.py · explain.py · visit_prep.py · privacy.py · vault.py        api/app.py · streamlit_app.py · frontend_example/
```

**Decision table** (`pipeline.decide`): labs beat text; text alone is used when there is no lab. Lab deficient + text denies → **conflicting**. Lab normal + text affirms → **conflicting**.
Weak markers (serum iron alone) → at most `suspected`. Several values of one test: separated only when the report itself justifies it (explicit "currently/now", or all values dated);
otherwise **conflicting**; older low values are kept in `historical_values` / `history_of_deficiency` (H03, H07).

**Safety rules in the laboratory extractor** (each has tests):
negative numbers (`-8`), scientific notation / non-finite (`1e999`, 400-digit strings), OCR-ambiguous tokens (`l8`, `1O`), decimal-comma numbers, missing units (provisional, never definitive),
implausible values (`Calcium 85 mg/dL`), medication doses (`vitamin D 2000 IU/day`), dates and reference ranges (`45 (ref 15–150)` → 45), unit mismatches.

**Statuses**: `deficient · borderline · suspected · not_deficient · not_assessed · conflicting`. **Reliability**: `ok · verify · abstain`.
Per-document verdict: `ok · verify · insufficient_information · cannot_determine`.

**Robustness limits**: 50,000 characters per text (longer → controlled rejection, API 422), 500 statements / 300 values / 500 dates analysed per document, bounded regexes
(adversarial repeated text at the limit finishes in about 2 s; an earlier version needed ~55 s and was fixed). Embedded instructions (`Ignore all rules…`) are ignored and flagged;
markup (`<script>`) is treated as plain text and flagged; the web UI and Streamlit never render report text as HTML.

---------------------------------------------------------------------------------------------------------------------------

## 3. The eight features

1. **Medical Report Detective** — `nutrient.detective = {evidence[], context[], missing_information[]}`. Each evidence item has the finding text, the source line, page, section heading, line,
   character span (for highlighting), date, cut-off used and verification flags. Context lists supplements mentioned *or denied*, symptoms, risk factors, history. Missing information lists
   absent reference range, date, unit, albumin (calcium), transferrin saturation (ferritin only), MMA/homocysteine (borderline B12), sex (hemoglobin).
2. **Health Timeline** — `POST /v1/records/analyze` → `timeline.series[]`: points with date, value, converted unit, status, quote and location; trend `decreasing / increasing / stable / fluctuating /
   insufficient_data`; flagged downward trends. Excluded values are listed with the reason (no date, ambiguous date, missing unit, `<`/`>` value, unverified). Numeric dates like `03/04/2025` are
   **never guessed**: pass `date_order: "DMY"|"MDY"` or the value is excluded. "A trend is not a diagnosis."
3. **Explain at my level** — `POST /v1/explain` with `level` = `simple_en · detailed_en · medical_en · te · hi`. Template-generated, each sentence cites `E1, E2…`; `verify_explanation` re-checks every
   number, quote and status against the analysis and reports tampering (tested).
4. **Contradiction / missing-info detector** — `crosscheck.issues[]` of types `statement_conflict, symptom_conflict, value_conflict, lab_vs_statement, within_report_conflict, missing_or_unclear`, each
   with evidence from both records and a clarification question. Different dates lower the severity. It never decides which record is right.
5. **Knows when not to predict** — `quality.checks` answers the five questions (readable text? values extracted correctly? units clear? language supported? enough facts?) with actions such as
   "Please verify Vitamin D (page 3, line 2)". Per-line OCR confidence below 0.90 is flagged; below 0.70 the value becomes provisional. Whole-document OCR confidence below 0.60 → `cannot_determine`.
   *Limit:* if OCR reads 18 as 78 with high confidence, text alone cannot reveal it; only confidence scores and (for you) comparing to the original image can.
6. **One-click document understanding** — `processing_report`: file format, document type (`laboratory report / clinical note / mixed`), language, extraction method, per-page status
   (`ok / needs_verification / failed`), and `values_to_verify` with page, line and quote.
7. **Doctor-visit preparation** — `POST /v1/visit-prep` → `markdown` + printable `html` (user text HTML-escaped): results table with dates/sources, changes over time, items to clarify, standard and
   finding-specific questions (English; the five standard questions also in Hindi/Telugu). It never recommends a supplement or dose.
8. **Privacy-first vault** — `/v1/vault/*` with header `X-Vault-Key` (≥ 16 characters; stored only as a hash). Nothing is stored without `consent: true`; the stored record is a **masked** summary
   (names, phones, IDs, e-mails, DOB by pattern); originals (optional) live in a separate folder; list / get / delete / delete-all / export timeline & data; automatic retention
   (`NUTRICHECK_RETENTION_DAYS`, default 90). `GET /v1/privacy/notice` explains where data is processed and how long it is kept.
   *Limits:* files are **not encrypted at rest** by this code (use disk encryption); masking is pattern-based and can miss unlabelled names/addresses; the key is a shared secret, not a full
   authentication system — put the API behind real auth/HTTPS for production.

---------------------------------------------------------------------------------------------------------------------------

## 4. Trained context classifier (verifier)

`python scripts/train_context_classifier.py` (seconds, no internet) → `models/context_clf.joblib`. TF-IDF char+word n-grams → logistic regression, 8 classes, trained on synthetic English
sentences generated from templates (train and test use disjoint cue phrases).

| Test | n | accuracy | macro-F1 |
|---|---|---|---|
| Unseen cue phrases (held out) | 2,400 | 78.2 % | 0.786 |
| Hand-written dev set | 120 | 90.0 % | 0.898 |
| Hand-written "blind" set | 120 | 95.0 % | 0.951 |

The two hand-written sets were both written by me in the same session (the "blind" one is blind to the generator's grammar, not to me) — treat them as optimistic.
Inside the system the classifier only **verifies**: on the 370 gold cases it changed 3 results to `conflicting` (all moved to human review, none became a confident wrong answer).
Switch it off with `use_classifier=false` / `NUTRICHECK_DISABLE_CLASSIFIER=1`; `NUTRICHECK_CLASSIFIER_POLICY=flag` only raises the flag.

**Real upgrade** (needs internet/GPU; not run here): `python scripts/finetune_transformer.py --model dmis-lab/biobert-base-cased-v1.2 --epochs 4 --out models/context_bert`
(also `emilyalsentzer/Bio_ClinicalBERT`), then `NUTRICHECK_CONTEXT_MODEL=models/context_bert`. Only a tiny offline model was run to prove the code path. Train on real, de-identified,
clinician-labelled English sentences, split by patient.

## 5. BioGPT (optional second opinion)

`biogpt_engine.py` (Hugging Face `microsoft/biogpt`, PyTorch). It never sees the raw report (so instructions inside a document cannot steer it): it receives an English summary built only from extracted
fields, answers a yes/no question per nutrient, and a disagreement only sets `needs_human_review` — **it never changes a status**. Enable with `use_biogpt=true` after
`pip install -r requirements-biogpt.txt`. Only a tiny random BioGPT was run here; BioGPT's real accuracy on your data is **not measured**.

## 6. API

`uvicorn nutricheck_nlp.api.app:app --app-dir src --reload` → `/docs`.

| Endpoint | Purpose |
|---|---|
| `GET /health`, `/v1/models/status`, `/v1/nutrients` | status, components, reference table |
| `POST /v1/analyze/text` · `/file` · `/batch` · `/image` | analysis (text, PDF/DOCX/CSV/TXT/image, up to 50 texts) |
| `POST /v1/records/analyze` · `/v1/records/analyze/files` | several reports → timeline + contradictions |
| `POST /v1/explain` | explanation at a chosen level/language |
| `POST /v1/visit-prep` | doctor-visit summary |
| `POST /v1/privacy/mask`, `GET /v1/privacy/notice` | masking, privacy notice |
| `POST/GET/DELETE /v1/vault/...` | vault (header `X-Vault-Key`) |

Upload safety: extension whitelist, MIME and magic-byte checks, size limit (`NUTRICHECK_MAX_UPLOAD_MB`, default 10), PDF page limit, corrupt/encrypted files → clear 4xx errors, in-memory processing.
Before going public set `NUTRICHECK_API_KEY` (call from your server, not browser JS), `NUTRICHECK_CORS_ORIGINS`, HTTPS and rate limiting.

## 7. Tests

`python run_all_tests.py`. Person 3 groups: gold v1 + v2 (370), H01–H24 hardening, the eight features (`tests/test_features.py`), synthetic, documents (real generated PDF/DOCX/CSV),
API, classifier, lab reference ranges, deploy/frontend (no HTML injection), PyTorch parts (skipped without torch). **Passing them is not clinical validation.**

## 8. Honest limitations

* Not run here (no internet / not installed): real BioBERT fine-tuning, real BioGPT weights, real PaddleOCR on English images/scans. The English OCR fixture in `tests/fixtures/` is **synthetic**.
* Telugu/Hindi wording is not clinician-reviewed.
* One adult threshold set (no age/pregnancy-specific ranges); the worst tier-1 iron marker decides; a number not adjacent to its analyte name is not linked to it.
* Comparisons across laboratories/methods can mislead; the timeline says so.
* Non-ASCII look-alike digits and unusual layouts can still fool any rule-based reader — that is what the verification actions are for.
* English only; other languages are refused, not translated.
