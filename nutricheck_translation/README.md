# NutriCheck Translation Module

This folder adds a separate translation stage between Person 1 OCR and Person 3 NLP.

## Flow

`Medical report -> Person 1 OCR -> translation -> Person 3 NLP -> results`

## Included support

- Portuguese (`pt`) OCR text is translated to English using `Helsinki-NLP/opus-mt-pt-en`.
- English (`en`) text passes through unchanged.
- Other languages are stopped with an explicit status; they are not silently sent to Person 3.
- The original OCR text is retained in every translation result.
- Common numeric values, dates, ranges and lab units are protected during translation where possible.

## Setup (Windows PowerShell)

Run these commands from the NutriCheck project root:

```powershell
# Activate your existing project environment if you have one:
.\.venv\Scripts\Activate.ps1

# Install translation dependencies:
pip install -r translation\requirements.txt
```

The first Portuguese translation downloads the MarianMT model from Hugging Face and may take time. A network connection is required for the first run. The model is cached for later runs.

## Quick test

From the NutriCheck project root:

```powershell
python -c "from translation import MedicalReportTranslator; t=MedicalReportTranslator(); r=t.translate_text('Hemoglobina: 10 g/dL', source_language='pt'); print(r.to_dict())"
```

## Connect to Person 3

Person 3's package should be importable from your integration environment. Example:

```python
from nutricheck_nlp import NutriCheckNLP
from translation import MedicalReportTranslator

translator = MedicalReportTranslator()
nlp = NutriCheckNLP()

ocr_text = "Hemoglobina: 10 g/dL\nVitamina D: 15 ng/mL"
translation = translator.translate_text(ocr_text, source_language="pt")

if translation.status in {"translated", "passed_through"}:
    result = nlp.analyze_text(translation.translated_text)
else:
    raise RuntimeError(
        f"Translation stage stopped: {translation.status}; "
        f"{translation.warnings}"
    )

print("Original OCR:", translation.original_text)
print("English text:", translation.translated_text)
print("NLP result:", result)
```

If you use `integration/pipeline.py`, add the `translation` folder to Python's import path by running the script from the project root, or make the project root your working directory.

## Important medical-safety limitations

This is a starter integration, not a clinically validated translation system. Do not use its output as the sole basis for diagnosis, treatment, or patient decisions. Always compare the translated text with the original OCR, especially:

- negation (e.g. “no evidence of”),
- abnormal/normal indicators,
- decimal commas versus decimal points,
- dates and reference ranges,
- units and analyte names.

Numeric token protection reduces the chance that values and units change, but it does not guarantee perfect preservation. Have a qualified person review translations before clinical use. Keep original OCR and translated text together for traceability.

## Tests

The unit tests do not download the translation model:

```powershell
python -m unittest discover -s tests -v
```

## Files

- `translation/translator.py` — translation class and result structure
- `translation/language_detector.py` — Portuguese/English language detection
- `translation/pipeline_adapter.py` — example bridge into Person 3
- `translation/requirements.txt` — Python dependencies
- `tests/test_translator.py` — basic tests
