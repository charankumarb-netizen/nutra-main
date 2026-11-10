"""
NutriCheck NLP pipeline

    text / OCR text  ->  language detection -> optional Romance-to-English translation
                     ->  English language guard -> labs + mentions + hints -> decision engine
                     ->  evidence + quality checks -> (optional) BioGPT second opinion

The clinical NLP rules remain English-only. Supported Romance-language text may be
translated to English first; unsupported languages and failed translations are refused.
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from . import __version__
from .biogpt_engine import BioGPTSecondOpinion
from .dates import find_dates, document_date
from .document_extractor import DocumentError, ExtractedDocument, extract_document
from .evidence import build_detective, fmt_date, location_text, quote_for, section_for
from .lab_extractor import LabExtractor, LabValue
from .language_guard import detect_language, unsupported_message
from .lexicon import ANALYTES, FINDING_KEYS, NUTRIENT_KEYS, TARGET_TIERS, TARGETS, display_unit
from .quality import OCR_LINE_MIN_CONFIDENCE, OCR_MIN_CONFIDENCE, assess_quality, reason_text
from .text_extractor import Mention, MentionExtractor
from .text_normalizer import MAX_TEXT_CHARS, normalize_text

DISCLAIMER = (
    "Screening-support tool. This is not a medical diagnosis; results must be confirmed by a healthcare professional."
)

STATUS_LABEL = {
    "deficient": "Deficiency detected",
    "borderline": "Borderline / insufficiency",
    "suspected": "Suspected (needs confirmation)",
    "not_deficient": "No deficiency",
    "not_assessed": "Not assessed",
    "conflicting": "Conflicting information (human review)",
}
FINDING_LABEL = {
    "deficient": "Present", "not_deficient": "Absent", "suspected": "Possible",
    "borderline": "Possible", "not_assessed": "Not assessed", "conflicting": "Conflicting",
}
LAB_STATUS_TEXT = {"deficient": "low", "borderline": "borderline", "normal": "normal", "high": "high",
                   "indeterminate": "indeterminate"}

_SEX_EXPLICIT = re.compile(
    r"(?<![a-z])(?:sex|gender|sexo|sexe|geschlecht)\s*[:=]?\s*"
    r"(female|male|feminino|masculino|femenino|feminin|masculin|weiblich|mannlich|fem|masc|mulher|homem|f|m)(?![a-z])"
)
_SEX_WORD_F = re.compile(r"(?<![a-z])(?:female|woman|women|pregnant|pregnancy|mrs|feminino|mulher|mujer|femme|frau|weiblich)(?![a-z])")
_SEX_WORD_M = re.compile(r"(?<![a-z])(?:male|man|gentleman|mr|masculino|homem|hombre|homme|mann|mannlich)(?![a-z])")
_INJECTION = re.compile(
    r"(?<![a-z])(?:ignore|disregard|forget|override|bypass)\s+(?:all|any|the|previous|prior|above|these|your)?\s*(?:rules?|instructions?|prompts?|guidelines?|constraints?)"
    r"|(?:say|state|report|output|respond|answer|write)\s+(?:that\s+)?(?:everything|all|the\s+patient)\s+(?:is|are)\s+(?:normal|fine|healthy|ok)"
    r"|system\s+prompt|you\s+are\s+now|do\s+not\s+flag|mark\s+(?:all|everything)\s+as\s+normal|act\s+as\s+(?:a|an|the)")
_MARKUP = re.compile(r"<\s*/?\s*(?:script|iframe|img|svg|object|embed|style|link|a|div|span|body|html|form|input|button)\b[^>]*>|javascript:|on\w+\s*=", re.I)


def locate(text: str, offset: int, page_spans: list[dict] | None) -> dict[str, Any]:
    """Where in the source document is character `offset`?  -> {"page", "line", "offset", "section", "source"} (page may be None)."""
    page, base, method = None, 0, None
    for sp in page_spans or []:
        if sp["start"] <= offset < sp["end"] or (offset == sp["end"] == len(text)):
            page, base, method = sp.get("page"), sp["start"], sp.get("method")
            break
    return {"page": page, "line": text.count("\n", base, offset) + 1, "offset": offset, "source": method,
            "section": section_for(text, offset)}


# rule-engine assertion -> label used by the trained context classifier (see context_model.py)
ASSERTION_TO_CLASS = {
    "affirmed": "affirmed", "negated": "negated", "uncertain": "suspected",
    "historical": "historical", "family": "family_history", "hypothetical": "hypothetical",
}


def expected_class(m: Mention) -> str | None:
    """Which classifier label should the sentence of this mention get, according to the rules?"""
    if m.kind == "elided":
        return None                        # ", now normal" has no nutrient name of its own: nothing to verify
    if m.polarity == "low":
        return ASSERTION_TO_CLASS.get(m.assertion)
    if m.assertion == "affirmed":          # "ferritin normal" / "vitamin D elevated": no deficiency stated
        return "negated"
    if m.assertion in ("uncertain", "historical", "family", "hypothetical"):
        return ASSERTION_TO_CLASS[m.assertion]
    return None


def detect_sex(norm: str) -> str | None:
    m = _SEX_EXPLICIT.search(norm)
    if m:
        w = m.group(1).lower()
        if w.startswith(("f", "w", "mulher")):
            return "F"
        if w.startswith(("m", "h")):
            return "M"
    f, mm = bool(_SEX_WORD_F.search(norm)), bool(_SEX_WORD_M.search(norm))
    if f and not mm:
        return "F"
    if mm and not f:
        return "M"
    return None


# ---------------------------------------------------------------------------
# Decision engine
# ---------------------------------------------------------------------------


def _date_key(d: dict | None) -> tuple[int, int, int] | None:
    if not d or not d.get("iso"):
        return None
    parts = [int(x) for x in d["iso"].split("-")]
    return (parts[0], parts[1] if len(parts) > 1 else 0, parts[2] if len(parts) > 2 else 0)


def resolve_same_analyte(labs: list[LabValue]) -> tuple[list[LabValue], list[LabValue], bool]:
    """Several values of ONE analyte -> (current values, older values, unresolved_conflict).

    Values are separated only when the report itself justifies it: an explicit "currently/now" marker, or every value
    carrying a (non-ambiguous) date, or an explicit "previously/earlier" marker.  Otherwise nothing is guessed.
    """
    if len(labs) < 2:
        return list(labs), [], False
    cur = [l for l in labs if l.temporal == "current"]
    if cur and len(cur) < len(labs):
        return cur, [l for l in labs if l not in cur], False
    keys = [_date_key(l.date) for l in labs]
    if all(k is not None for k in keys) and len(set(keys)) > 1:
        latest = max(keys)
        return [l for l, k in zip(labs, keys) if k == latest], [l for l, k in zip(labs, keys) if k != latest], False
    past = [l for l in labs if l.temporal == "past"]
    if past and len(past) < len(labs):
        return [l for l in labs if l not in past], past, False
    return list(labs), [], len({l.status for l in labs}) > 1


def _lab_status(target: str, labs: list[LabValue]) -> dict[str, Any]:
    """lab_status: deficient|borderline|suspected|normal|normal_weak|conflict|none (+ the labs that decided and the older ones)."""
    notes: list[str] = []
    reasons: list[str] = []
    mine = [l for l in labs if l.target == target and l.usable]
    firm = [l for l in mine if not l.provisional]
    prov = [l for l in mine if l.provisional]
    decided: tuple[str, list[LabValue]] | None = None
    older: list[LabValue] = []
    for tier in TARGET_TIERS[target]:
        tier_labs = [l for l in firm if l.analyte in tier]
        if not tier_labs:
            continue
        current: list[LabValue] = []
        conflict = False
        for an in tier:
            cur, old, c = resolve_same_analyte([l for l in tier_labs if l.analyte == an])
            current += cur
            older += old
            conflict = conflict or c
        weak = all(l.weak for l in current)
        stats = {l.status for l in current}
        if conflict:
            st = "conflict"
            reasons.append("conflicting_measurements")
            notes.append("The report contains different values of the same test without dates or wording that say which is current: "
                         + ", ".join(f"{l.value:g} {l.unit}" for l in current) + ".")
        elif "deficient" in stats:
            st = "suspected" if weak else "deficient"
        elif "borderline" in stats:
            st = "borderline"
        else:
            st = "normal_weak" if weak else "normal"
        if decided is None:
            decided = (st, current)
            if "high" in stats and st == "normal":
                notes.append("Value above the reference range (this does not indicate a deficiency).")
        else:
            if "deficient" in stats and decided[0] in ("normal", "normal_weak"):
                notes.append("A lower-weight marker (" + ", ".join(sorted({l.label for l in current})) +
                             ") is low, but the primary marker is normal.")
    if decided is None and prov:
        low = any(l.status in ("deficient", "borderline") for l in prov)
        reasons.append("unit_missing")
        notes.append("The only value(s) found have a missing unit, so they are provisional and cannot support a definitive result.")
        return {"status": "suspected" if low else "none", "used": prov, "older": older, "notes": notes, "reasons": reasons}
    if decided is not None and prov:
        reasons.append("unit_missing")
        notes.append("An additional value without a unit was not used for the decision.")
    if decided is None:
        return {"status": "none", "used": [], "older": older, "notes": notes, "reasons": reasons}
    return {"status": decided[0], "used": decided[1], "older": older, "notes": notes, "reasons": reasons}


def _text_state(target: str, mentions: list[Mention]) -> tuple[str, str, list[Mention]]:
    """Return (state, level, relevant mentions); state: affirmed|denied|conflict|uncertain|historical|none"""
    rel = [m for m in mentions if m.target == target and m.stance != "ignored"]
    aff = [m for m in rel if m.stance == "affirmed"]
    den = [m for m in rel if m.stance == "denied"]
    unc = [m for m in rel if m.stance == "uncertain"]
    his = [m for m in rel if m.stance == "historical"]
    level = "deficient" if any(m.level == "deficient" for m in aff) else "borderline"
    if aff and den:
        return "conflict", level, rel
    if aff:
        return "affirmed", level, rel
    if den:
        return "denied", level, rel
    if unc:
        return "uncertain", level, rel
    if his:
        return "historical", level, rel
    return "none", level, rel


def decide(lab_status: str, text_state: str, text_level: str) -> tuple[str, float, list[str]]:
    """Core decision table -> (status, confidence, warnings)."""
    w: list[str] = []
    L, T = lab_status, text_state
    if L == "conflict":
        return "conflicting", 0.40, w
    if L == "deficient":
        if T == "affirmed":
            return "deficient", 0.99, w
        if T in ("denied", "conflict"):
            w.append("The text contradicts the laboratory result. Neither was silently preferred: please have this clarified.")
            return "conflicting", 0.50, w
        return "deficient", 0.97, w
    if L == "borderline":
        if T == "affirmed" and text_level == "deficient":
            return "deficient", 0.85, w
        if T in ("denied", "conflict"):
            w.append("The text contradicts the laboratory result (borderline value).")
            return "borderline", 0.75, w
        return "borderline", 0.95, w
    if L == "suspected":
        if T == "affirmed":
            return "deficient", 0.90, w
        if T == "denied":
            return "not_deficient", 0.70, w
        return "suspected", 0.60, w
    if L == "normal":
        if T in ("affirmed", "conflict"):
            w.append("The text states a deficiency but the laboratory result is normal (treatment under way, an old statement, or an error?).")
            return "conflicting", 0.50, w
        return "not_deficient", 0.97, w
    if L == "normal_weak":
        if T == "affirmed":
            return "deficient", 0.85, w
        if T == "denied":
            return "not_deficient", 0.80, w
        return "not_deficient", 0.60, w
    # no lab
    if T == "affirmed":
        return ("borderline", 0.85, w) if text_level == "borderline" else ("deficient", 0.90, w)
    if T == "conflict":
        w.append("The text contains contradictory statements.")
        return "conflicting", 0.40, w
    if T == "denied":
        return "not_deficient", 0.85, w
    if T == "uncertain":
        return "suspected", 0.60, w
    return "not_assessed", 0.0, w


def _lab_evidence(l: LabValue) -> str:
    cmp_ = l.comparator or ""
    a = ANALYTES[l.analyte]
    ref = f"< {a.deficient_below:g}" if a.deficient_below is not None else ""
    if a.borderline_below is not None:
        ref = f"< {a.deficient_below:g} deficient; {a.deficient_below:g}-{a.borderline_below:g} borderline"
    when = f" [{fmt_date(l.date)}]" if l.date else ""
    return (f"{l.label}: {cmp_}{l.value:g} {display_unit(l.unit)} → {LAB_STATUS_TEXT.get(l.status, l.status)}{when} "
            f"(reference: {ref} {display_unit(a.canonical_unit)})")


def _map_ocr_lines(text: str, ocr_lines: list[dict] | None) -> list[tuple[int, int, float]]:
    out: list[tuple[int, int, float]] = []
    pos = 0
    for ln in ocr_lines or []:
        t, c = str(ln.get("text", "")), ln.get("confidence")
        if not t or c is None:
            continue
        i = text.find(t, pos)
        if i < 0:
            continue
        out.append((i, i + len(t), float(c)))
        pos = i + len(t)
    return out


# ---------------------------------------------------------------------------


class NutriCheckNLP:
    """High-level entry point used by the API, Streamlit app and tests.

    Parameters
    ----------
    biogpt      optional BioGPTSecondOpinion (never changes a status, only raises review flags)
    ocr_engine  optional object with process_image(path) -> dict (Person 1's MedicalOCREngine);
                when omitted it is created lazily from ../person1-ocr/src the first time it is needed
    classifier  optional trained sentence classifier (context_model.py): a *verifier*, not a decider
    classifier_policy  "conflict" (default): a high-confidence disagreement on a text-only decision
                makes the status `conflicting`; "flag": only raise needs_human_review
    classifier_threshold  minimum model probability for a disagreement to count (default 0.85)
    """

    def __init__(
        self,
        biogpt: BioGPTSecondOpinion | None = None,
        ocr_engine: Any = None,
        classifier: Any = None,
        classifier_policy: str = "conflict",
        classifier_threshold: float = 0.85,
        translator: Any = None,
    ) -> None:
        if classifier_policy not in ("conflict", "flag"):
            raise ValueError("classifier_policy must be 'conflict' or 'flag'")
        self.mentions = MentionExtractor()
        self.biogpt = biogpt
        self.classifier = classifier
        self.classifier_policy = classifier_policy
        self.classifier_threshold = classifier_threshold
        self.translator = translator
        self._ocr = ocr_engine
        self._ocr_error: str | None = None

    # ------------------------------------------------------------------
    def analyze_text(
        self,
        text: str,
        sex: str | None = None,
        ocr_confidence: float | None = None,
        use_biogpt: bool = False,
        explain: bool = False,
        page_spans: list[dict] | None = None,
        use_classifier: bool | None = None,
        date_order: str | None = None,
        ocr_lines: list[dict] | None = None,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        if isinstance(text, str) and len(text) > MAX_TEXT_CHARS:
            return self._input_rejected(
                f"The text is longer than the {MAX_TEXT_CHARS:,}-character limit ({len(text):,} characters). "
                "Split the report and analyse the parts separately.", "text_too_long", t0)
        original = text if isinstance(text, str) else ""
        original = original.replace("\x00", " ")
        norm = normalize_text(original)
        warnings: list[str] = []
        security_flags: list[str] = []
        needs_review = False

        if not norm.strip():
            warnings.append("The text is empty.")
        if _INJECTION.search(norm):
            security_flags.append("instruction_like_text")
            warnings.append("Instruction-like text was found inside the document and was IGNORED: only clinical content is analysed.")
        if _MARKUP.search(original):
            security_flags.append("markup_in_text")
            warnings.append("HTML/script markup was found in the text. It is treated as plain text and must be escaped when displayed.")

        # Keep the source report separate from the text used by the English NLP rules.
        source_text = original
        translation_info: dict[str, Any] | None = None
        was_translated = False

        lang = detect_language(norm, original)
        supported = lang["supported"]

        # The NLP rules remain English-only. Translate only languages supported by
        # the configured OPUS ROMANCE -> English translator.
        translatable_languages = {"pt", "es", "fr", "it"}
        if not supported and lang["code"] in translatable_languages:
            try:
                if self.translator is None:
                    project_root = Path(__file__).resolve().parents[3]
                    if str(project_root) not in sys.path:
                        sys.path.insert(0, str(project_root))
                    from translation.translator import MedicalReportTranslator
                    self.translator = MedicalReportTranslator()

                translation_result = self.translator.translate_text(
                    source_text, source_language=lang["code"]
                )
                translation_info = translation_result.to_dict()

                if translation_result.status == "translated" and translation_result.translated_text.strip():
                    original = translation_result.translated_text.strip()
                    norm = normalize_text(original)
                    lang = detect_language(norm, original)
                    supported = lang["supported"]
                    was_translated = supported
                    if supported:
                        warnings.extend(translation_result.warnings)
                        warnings.append(
                            "This report was translated before NLP analysis. "
                            "Verify every finding against the original-language report."
                        )
                        needs_review = True
                    else:
                        warnings.append(
                            "The translated text could not be confirmed as English; no clinical assessment was made."
                        )
                elif translation_result.status == "passed_through":
                    # Defensive handling for injected translators that pass text through.
                    original = translation_result.translated_text
                    norm = normalize_text(original)
                    lang = detect_language(norm, original)
                    supported = lang["supported"]
                    was_translated = False
                else:
                    warnings.extend(translation_result.warnings)
                    warnings.append(
                        "Translation was unsuccessful. No clinical assessment was made from the source text."
                    )
            except Exception as exc:
                translation_info = {
                    "original_text": source_text,
                    "translated_text": "",
                    "source_language": lang["code"],
                    "target_language": "en",
                    "status": "translation_error",
                    "warnings": [f"Translation failed: {type(exc).__name__}: {exc}"],
                }
                warnings.append(
                    "Translation failed. No clinical assessment was made from the source text."
                )

        if was_translated:
            # Re-check translated content as untrusted document text.
            if _INJECTION.search(norm):
                if "instruction_like_text" not in security_flags:
                    security_flags.append("instruction_like_text")
                warnings.append(
                    "Instruction-like text was found in the translated document and was IGNORED."
                )
            if _MARKUP.search(original):
                if "markup_in_text" not in security_flags:
                    security_flags.append("markup_in_text")
                warnings.append(
                    "Markup was found in the translated document and is treated as plain text."
                )

            # Character offsets from the source report do not apply to translated text.
            # Keep aggregate OCR confidence, but disable source-line/page offset mapping.
            page_spans = None
            ocr_lines = None

        if not supported:
            warnings.append(unsupported_message(lang))
        if sex not in (None, "M", "F"):
            sex = None
        sex_source = "provided" if sex else None
        if sex is None and supported:
            sex = detect_sex(norm)
            sex_source = "detected" if sex else None

        if supported:
            labs, rejected = LabExtractor(sex, date_order).extract(norm, original)
            mentions = self.mentions.extract(norm, original)
            if len(mentions) >= MentionExtractor.MAX_MENTIONS:
                warnings.append(f"Very many statements were found; only the first {MentionExtractor.MAX_MENTIONS} were analysed. Please review the report manually.")
                needs_review = True
            symptoms = self.mentions.symptoms(norm)
            symptom_statements = self.mentions.symptom_statements(norm)
            risks = self.mentions.risk_factors(norm)
            treatment_statements = self.mentions.treatments(norm, original, include_negated=True)
            treatments = [t for t in treatment_statements if t["assertion"] != "negated"]
            for st in symptom_statements:
                s0, e0 = self.mentions._sentence_bounds(norm, st["start"])
                st["sentence"] = original[s0:e0].strip()
                st["location"] = locate(original, st["start"], page_spans)
        else:
            labs, rejected, mentions, symptoms, symptom_statements, risks, treatment_statements, treatments = [], [], [], [], [], [], [], []

        date_hits = find_dates(norm, date_order) if supported else []
        doc_date_hit = document_date(norm, date_hits) if supported else None

        # per-line OCR confidence -> verification flags (a low-confidence line must never be trusted silently)
        conf_ranges = _map_ocr_lines(original, ocr_lines)
        low_conf_lines: list[dict] = []
        for rng in conf_ranges:
            if rng[2] < OCR_LINE_MIN_CONFIDENCE:
                low_conf_lines.append({"start": rng[0], "end": rng[1], "confidence": round(rng[2], 3)})
        for l in labs:
            for s0, e0, c in conf_ranges:
                if s0 <= l.start < e0 and c < OCR_LINE_MIN_CONFIDENCE:
                    l.verify_reasons.append("low_ocr_confidence")
                    l.warnings.append(f"This line was read by OCR with low confidence ({c:.2f}); check the original image.")
                    if c < 0.70:
                        l.provisional = True
                    break

        for l in labs:
            warnings.extend(f"{l.label}: {w}" for w in l.warnings if "Sex not stated" not in w)
        if any("Sex not stated" in w for l in labs for w in l.warnings):
            warnings.append("Sex not stated: the 12 g/dL anemia threshold was used.")
        for r in rejected:
            r["location"] = locate(original, r["start"], page_spans)
            r["location_text"] = location_text(r["location"], r["location"].get("section"))
            if r["reason"].startswith("unit_mismatch"):
                warnings.append(f"Unit not compatible with {r['analyte']} ({r['raw_text']}); value ignored.")
            elif r["reason"] not in ("date_or_reference_range",):
                warnings.append(f"Invalid value rejected ({r['raw_text']}): {reason_text(r['reason'])}.")

        if ocr_confidence is not None and ocr_confidence < OCR_MIN_CONFIDENCE:
            warnings.append(f"Low OCR confidence ({ocr_confidence:.2f}): verify the values against the document.")
            needs_review = True
        if low_conf_lines:
            needs_review = True
        if any(l.verify_reasons for l in labs):
            needs_review = True
        if any(r["reason"] not in ("date_or_reference_range",) for r in rejected):
            needs_review = True
        if any(l.ref_conflict for l in labs):
            needs_review = True
        if not supported:
            needs_review = True

        # -- sentence + location for every mention (evidence highlighting) ----------------------
        sentences: dict[int, str] = {}
        for i, m in enumerate(mentions):
            s0, e0 = self.mentions._sentence_bounds(norm, m.start)
            sentences[i] = original[s0:e0].strip()

        # -- trained classifier = verifier of the rule engine's context decision -----------------
        classifier_report: dict[str, Any] | None = None
        clf_flags: dict[str, list[dict]] = {}
        clf_by_mention: dict[int, dict] = {}
        want_clf = use_classifier is not False and self.classifier is not None and supported
        if use_classifier is True and self.classifier is None:
            warnings.append("The trained classifier is unavailable: the result is based on the rules only.")
        if want_clf and mentions:
            try:
                clauses = []
                for m in mentions:
                    c0, c1 = self.mentions.clause_bounds(norm, m.start, m.end)
                    clauses.append(original[c0:c1].strip())
                preds = self.classifier.predict(clauses)
            except Exception as exc:  # a broken model must never break the analysis
                preds = None
                warnings.append(f"The trained classifier failed ({type(exc).__name__}); the result is based on the rules only.")
            if preds is not None:
                items = []
                for i, (m, (label, prob)) in enumerate(zip(mentions, preds)):
                    exp = expected_class(m)
                    agrees = exp is None or label == exp or prob < self.classifier_threshold
                    rec = {"target": m.target, "sentence": clauses[i], "rule_label": exp, "model_label": label,
                           "probability": round(float(prob), 4), "agrees": agrees}
                    clf_by_mention[i] = rec
                    items.append(rec)
                    if not agrees:
                        rec["decisive"] = m.stance != "ignored"
                        clf_flags.setdefault(m.target, []).append(rec)
                classifier_report = {
                    "available": True, "name": getattr(self.classifier, "name", "classifier"),
                    "policy": self.classifier_policy, "threshold": self.classifier_threshold,
                    "disagreements": sorted(clf_flags), "items": items,
                }
        elif want_clf:
            classifier_report = {"available": True, "name": getattr(self.classifier, "name", "classifier"),
                                 "policy": self.classifier_policy, "threshold": self.classifier_threshold,
                                 "disagreements": [], "items": []}

        mention_idx = {id(m): i for i, m in enumerate(mentions)}
        labs_out = [self._lab_out(l, original, page_spans) for l in labs]
        lab_out_by_id = {id(l): lo for l, lo in zip(labs, labs_out)}
        mention_out = []
        for i, m in enumerate(mentions):
            d = m.to_dict()
            d["sentence"] = sentences[i]
            d["location"] = locate(original, m.start, page_spans)
            if i in clf_by_mention:
                d["classifier"] = {k: clf_by_mention[i][k] for k in ("model_label", "probability", "agrees")}
            mention_out.append(d)

        nutrients, findings = [], []
        for key in list(NUTRIENT_KEYS) + list(FINDING_KEYS):
            tgt = TARGETS[key]
            ls = _lab_status(key, labs)
            lab_st, used, lab_notes = ls["status"], ls["used"], list(ls["notes"])
            t_state, t_level, rel = _text_state(key, mentions)
            status, conf, w = decide(lab_st, t_state, t_level)
            reasons: list[str] = list(ls["reasons"])
            if any(l.assumed_unit for l in used):
                conf = min(conf, 0.5)
            if any("low_ocr_confidence" in l.verify_reasons for l in used):
                conf = min(conf, 0.7)
            if status == "conflicting":
                reasons.append("conflicting_evidence")
            if status == "suspected":
                reasons.append("suspected_needs_confirmation")
            if lab_st in ("deficient",) and t_state in ("denied", "conflict") or (lab_st == "normal" and t_state in ("affirmed", "conflict")):
                reasons.append("text_contradicts_lab")
            if any(l.ref_conflict for l in used):
                reasons.append("lab_reference_disagreement")
            my_labs = [l for l in labs if l.target == key]
            for l in my_labs:
                for code in l.verify_reasons:
                    if code != "unit_missing" or l in used or not used:
                        reasons.append(code)
            for r in rejected:
                if r.get("target") == key and r["reason"] not in ("date_or_reference_range",):
                    reasons.append("unit_mismatch" if r["reason"].startswith("unit_mismatch") else r["reason"])
            if ocr_confidence is not None and ocr_confidence < OCR_MIN_CONFIDENCE and (my_labs or rel):
                reasons.append("low_ocr_confidence")
            if not supported:
                reasons = ["unsupported_language"]
            cd = clf_flags.get(key, [])
            if cd:
                reasons.append("classifier_disagreement")
                w.append("The trained classifier disagrees with the rules about the context: " + "; ".join(
                    f"rule={d['rule_label']}, model={d['model_label']} (p={d['probability']:.2f})" for d in cd))
                if (self.classifier_policy == "conflict" and lab_st == "none"
                        and any(d["decisive"] for d in cd) and status != "not_assessed"):
                    status, conf = "conflicting", min(conf, 0.40)
                    reasons.append("conflicting_evidence")
            warnings.extend(f"{tgt.name}: {x}" for x in w)
            reasons = list(dict.fromkeys(reasons))
            if reasons:
                needs_review = True

            # reliability: ok | verify | abstain
            data_problem = {"unit_missing", "ambiguous_ocr_characters", "ambiguous_decimal_format", "implausible_value",
                            "negative_value", "scientific_notation_or_non_finite", "unreadable_number", "unparseable_number",
                            "unit_mismatch", "unsupported_language", "comparator_not_decisive"}
            if not supported or (status == "not_assessed" and any(r in data_problem or r.startswith("unit_mismatch") for r in reasons)):
                reliability = "abstain"
            elif status in ("conflicting",) or (status == "suspected" and "unit_missing" in reasons):
                reliability = "abstain" if status == "conflicting" and "conflicting_measurements" in reasons else "verify"
            elif reasons:
                reliability = "verify"
            else:
                reliability = "ok"
            abstain_reasons = [r for r in reasons if reliability == "abstain" or r in data_problem]

            evidence = [_lab_evidence(l) for l in used] + [f"Text: “{m.text}” ({m.assertion})" for m in rel]
            evidence_items = [
                {**{k: lab_out_by_id[id(l)][k] for k in ("analyte", "label", "value", "comparator", "unit", "canonical_value",
                                                          "canonical_unit", "status", "reference_check", "lab_reference", "location", "date", "temporal")},
                 "type": "lab", "text": l.raw_text}
                for l in used
            ] + [
                {"type": "text", "text": m.text, "sentence": sentences[mention_idx[id(m)]], "assertion": m.assertion,
                 "stance": m.stance, "polarity": m.polarity, "location": mention_out[mention_idx[id(m)]]["location"]}
                for m in rel
            ]
            history = any(m.stance == "historical" for m in rel) or any(l.status in ("deficient", "borderline") for l in ls["older"])
            used_out = [lab_out_by_id[id(l)] for l in used]
            other_out = [lab_out_by_id[id(l)] for l in my_labs if l not in used]
            detective = build_detective(
                key, used_out, other_out,
                [mention_out[mention_idx[id(m)]] for m in rel],
                treatment_statements, symptom_statements, risks, original, sex, history, rejected,
            )
            item = {
                "key": key,
                "name": tgt.name,
                "status": status,
                "status_label": (FINDING_LABEL if tgt.kind == "finding" else STATUS_LABEL)[status],
                "confidence": round(conf, 3),
                "reliability": reliability,
                "abstain_reasons": abstain_reasons,
                "lab_status": lab_st,
                "text_status": t_state,
                "evidence": evidence,
                "evidence_items": evidence_items,
                "detective": detective,
                "notes": lab_notes,
                "history_of_deficiency": history,
                "historical_values": [{"value": l.value, "unit": l.unit, "status": l.status, "date": fmt_date(l.date)} for l in ls["older"]],
                "under_treatment": any(m.under_treatment for m in rel),
                "needs_review": bool(reasons),
                "review_reasons": reasons,
                "recommendation": tgt.recommendation if status in ("deficient", "borderline", "suspected", "conflicting") else "",
            }
            (nutrients if tgt.kind == "nutrient" else findings).append(item)

        # suggestions
        suggestions: list[str] = []
        by_key = {n["key"]: n for n in nutrients}
        anemia = next((f for f in findings if f["key"] == "anemia"), None)
        if anemia and anemia["status"] == "deficient" and by_key["iron"]["status"] == "not_assessed":
            suggestions.append("Anemia without an iron assessment: consider ferritin and transferrin saturation.")
        if anemia and anemia["status"] == "deficient" and by_key["vitamin_b12"]["status"] == "not_assessed":
            suggestions.append("Anemia without a B12/folate assessment: consider measuring them.")
        for s in symptoms:
            pending = [TARGETS[t].name for t in s["targets"] if t in by_key and by_key[t]["status"] == "not_assessed"]
            if pending:
                suggestions.append(f"Symptom ‘{s['label']}’ suggests assessing: {', '.join(pending)}.")

        summary = {st: [n["key"] for n in nutrients if n["status"] == st] for st in STATUS_LABEL}
        n_words = len(re.findall(r"[a-z]+", norm))
        if not supported:
            doc_type = "unsupported language"
        elif labs and len(labs) >= max(1, n_words // 12):
            doc_type = "laboratory report"
        elif labs and mentions:
            doc_type = "mixed (laboratory values and clinical text)"
        elif mentions or symptoms or risks:
            doc_type = "clinical note"
        else:
            doc_type = "no relevant clinical content found"

        if was_translated:
            # Translation is not clinically validated: require manual verification
            # for every result without changing the rule-engine status itself.
            for item in nutrients + findings:
                item["needs_review"] = True
                item["review_reasons"] = list(dict.fromkeys(
                    item["review_reasons"] + ["translation_review"]
                ))
                if item["reliability"] == "ok":
                    item["reliability"] = "verify"

        quality = assess_quality(
            text=original, language=lang, labs=labs_out, rejected=rejected, mentions=mention_out,
            ocr_confidence=ocr_confidence, low_conf_lines=low_conf_lines, page_report=None,
            nutrients=nutrients + findings,
        )
        if quality["verdict"] != "ok":
            needs_review = True

        result: dict[str, Any] = {
            "version": __version__,
            "language": lang["code"],
            "language_info": lang,
            "language_supported": supported,
            "translation": translation_info,
            "source_text": source_text if was_translated else None,
            "analysis_text": original if was_translated else None,
            "document_profile": {"type": doc_type, "lab_values": len(labs), "statements": len(mentions),
                                 "characters": len(original), "language": lang["name"]},
            "patient": {"sex": sex, "sex_source": sex_source},
            "nutrients": nutrients,
            "findings": findings,
            "summary": summary,
            "labs": labs_out,
            "rejected_values": rejected,
            "mentions": mention_out,
            "supporting_symptoms": symptoms,
            "symptom_statements": symptom_statements,
            "risk_factors": risks,
            "treatments": treatments,
            "treatment_statements": treatment_statements,
            "dates": {"found": [h.to_dict() for h in date_hits],
                      "document_date": ({**doc_date_hit.to_dict()} if doc_date_hit else None)},
            "suggestions": suggestions,
            "warnings": list(dict.fromkeys(warnings)),
            "needs_human_review": needs_review,
            "quality": quality,
            "security": {"flags": security_flags},
            "disclaimer": DISCLAIMER,
            "biogpt": None,
            "classifier": classifier_report,
        }

        if use_biogpt and supported:
            if self.biogpt is None:
                self.biogpt = BioGPTSecondOpinion()
            review = self.biogpt.review(result, explain=explain)
            result["biogpt"] = review
            if not review.get("available"):
                result["warnings"].append("BioGPT is unavailable: the result is based on the rules only.")
            elif review.get("disagreements"):
                result["needs_human_review"] = True
                result["warnings"].append("BioGPT disagrees with the rules for: " + ", ".join(review["disagreements"]) + ".")
                for n in nutrients:
                    if n["key"] in review["disagreements"]:
                        n["needs_review"] = True
                        n["review_reasons"] = list(dict.fromkeys(n["review_reasons"] + ["biogpt_disagreement"]))
        result["processing_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return result

    # ------------------------------------------------------------------
    def _input_rejected(self, message: str, code: str, t0: float) -> dict[str, Any]:
        """A structurally complete result for input that is refused before analysis (too long, etc.)."""
        res = self.analyze_text("", ocr_confidence=None)
        res["warnings"] = [message]
        res["needs_human_review"] = True
        res["input_rejected"] = {"code": code, "message": message}
        res["quality"]["verdict"] = "cannot_determine"
        res["quality"]["message"] = message
        res["processing_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return res

    @staticmethod
    def _lab_out(l: LabValue, original: str = "", page_spans: list[dict] | None = None) -> dict:
        loc = locate(original, l.start, page_spans)
        return {
            "analyte": l.analyte, "target": l.target, "label": l.label, "raw_text": l.raw_text,
            "value": l.value, "comparator": l.comparator, "unit": l.unit,
            "canonical_value": l.canonical_value, "canonical_unit": l.canonical_unit,
            "status": l.status, "assumed_unit": l.assumed_unit, "usable": l.usable, "provisional": l.provisional,
            "warnings": l.warnings, "lab_reference": l.lab_reference, "reference_check": l.reference_check,
            "verify_reasons": l.verify_reasons, "ocr_candidates": l.ocr_candidates,
            "date": l.date, "temporal": l.temporal, "start": l.start, "end": l.end,
            "location": loc, "section": loc.get("section"),
            "quote": quote_for(original, l.start, l.end),
        }

    # ------------------------------------------------------------------
    # OCR (Person 1)
    # ------------------------------------------------------------------
    def _get_ocr(self):
        """Return Person 1's OCR engine (created once). Raises RuntimeError when unavailable."""
        if self._ocr is not None:
            return self._ocr
        if self._ocr_error is not None:
            raise RuntimeError(self._ocr_error)
        candidates = []
        if os.getenv("NUTRICHECK_OCR_SRC"):
            candidates.append(Path(os.environ["NUTRICHECK_OCR_SRC"]))
        candidates.append(Path(__file__).resolve().parents[3] / "person1-ocr" / "src")
        for c in candidates:
            if c.is_dir() and str(c) not in sys.path:
                sys.path.insert(0, str(c))
        try:
            from ocr_engine import MedicalOCREngine  # type: ignore
            self._ocr = MedicalOCREngine(language="en")
        except Exception as exc:
            self._ocr_error = "OCR engine unavailable (install requirements-ocr.txt): " + str(exc)
            raise RuntimeError(self._ocr_error) from exc
        return self._ocr

    def _ocr_for_documents(self):
        """Object with process_image(path) that loads the OCR lazily and reports failures as DocumentError."""
        outer = self

        class _Lazy:
            def process_image(self, path):
                try:
                    return outer._get_ocr().process_image(path)
                except RuntimeError as exc:
                    raise DocumentError(str(exc), 503) from exc

        return self._ocr if self._ocr is not None else _Lazy()

    def _ocr_page_callback(self):
        """png bytes -> OCR dict, used for scanned PDF pages. Writes a temp file and always deletes it."""
        import tempfile

        engine = self._ocr_for_documents()

        def run(png: bytes):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(png)
                path = tmp.name
            try:
                return engine.process_image(path)
            finally:
                Path(path).unlink(missing_ok=True)

        return run

    def analyze_ocr_result(self, ocr_result: dict, **kw) -> dict[str, Any]:
        """Accepts the dict returned by Person 1's MedicalOCREngine."""
        if not isinstance(ocr_result, dict) or not ocr_result.get("success", False):
            err = (ocr_result or {}).get("error", "OCR failed") if isinstance(ocr_result, dict) else "invalid OCR result"
            res = self.analyze_text("", **kw)
            res["warnings"].append(f"OCR was not successful: {err}")
            res["needs_human_review"] = True
            return res
        kw.setdefault("ocr_confidence", ocr_result.get("average_confidence"))
        text = ocr_result.get("text", "")
        text = text if isinstance(text, str) else ""
        kw.setdefault("page_spans", [{"page": 1, "start": 0, "end": len(text), "method": "ocr"}])
        kw.setdefault("ocr_lines", ocr_result.get("lines"))
        res = self.analyze_text(text, **kw)
        res["ocr"] = {
            "average_confidence": ocr_result.get("average_confidence"),
            "line_count": ocr_result.get("line_count"),
            "image": Path(str(ocr_result.get("image", ""))).name,
        }
        return res

    def analyze_image(self, image_path: str | Path, **kw) -> dict[str, Any]:
        """Run Person 1's OCR (PaddleOCR) and then the NLP pipeline."""
        return self.analyze_ocr_result(self._get_ocr().process_image(image_path), **kw)

    # ------------------------------------------------------------------
    # Documents (PDF / DOCX / CSV / TXT / image)
    # ------------------------------------------------------------------
    def analyze_document(self, doc: ExtractedDocument, **kw) -> list[dict[str, Any]]:
        """Analyse every record of an extracted document; each result carries its source information."""
        results: list[dict[str, Any]] = []
        for rec in doc.records:
            kw_rec = dict(kw)
            conf = kw_rec.pop("ocr_confidence", None)
            if doc.ocr_confidence is not None:
                conf = doc.ocr_confidence if conf is None else min(conf, doc.ocr_confidence)
            kw_rec.setdefault("ocr_lines", getattr(doc, "ocr_lines", None) or None)
            res = self.analyze_text(rec.text, ocr_confidence=conf, page_spans=rec.pages, **kw_rec)
            res["record_id"] = rec.record_id
            res["document"] = {k: v for k, v in doc.meta().items() if k != "warnings"}
            for w in doc.warnings:
                if w not in res["warnings"]:
                    res["warnings"].append(w)
            if doc.needs_review or not rec.text.strip() or any("truncated" in w.lower() for w in doc.warnings):
                res["needs_human_review"] = True
                if not rec.text.strip():
                    res["warnings"].append("No analysable text in this record.")
            if doc.ocr_confidence is not None:
                res["ocr"] = {"average_confidence": doc.ocr_confidence, "line_count": None, "image": doc.filename}
            res["processing_report"] = self._processing_report(doc, rec, res)
            pr = res["processing_report"]
            res["quality"]["pages_needing_attention"] = [p["page"] for p in pr["pages"] if p["status"] != "ok"]
            if res["quality"]["pages_needing_attention"] and res["quality"]["verdict"] == "ok":
                res["quality"]["verdict"] = "verify"
                res["quality"]["message"] = "Some pages could not be processed cleanly; verify them against the original."
                res["needs_human_review"] = True
            results.append(res)
        return results

    @staticmethod
    def _processing_report(doc: ExtractedDocument, rec, res: dict) -> dict[str, Any]:
        """Which pages were processed successfully and which values need user verification (one-click document understanding)."""
        pages = []
        info = getattr(doc, "page_info", None) or []
        by_page = {p["page"]: p for p in info}
        spans = rec.pages or [{"page": None, "start": 0, "end": len(rec.text), "method": "text"}]
        for sp in spans:
            pno = sp.get("page")
            pi = by_page.get(pno, {})
            labs_here = [l for l in res["labs"] if l["location"].get("page") == pno]
            need = [l for l in labs_here if l["verify_reasons"] or not l["usable"]]
            chars = sp["end"] - sp["start"]
            status = "ok"
            notes = list(pi.get("notes", []))
            if chars == 0 or pi.get("status") == "failed":
                status = "failed"
            elif pi.get("status") == "needs_verification" or need or (pi.get("ocr_confidence") is not None and pi["ocr_confidence"] < OCR_MIN_CONFIDENCE):
                status = "needs_verification"
            pages.append({"page": pno, "method": sp.get("method") or pi.get("method"), "characters": chars,
                          "ocr_confidence": pi.get("ocr_confidence"), "values_found": len(labs_here),
                          "values_needing_verification": len(need), "status": status, "notes": notes})
        verify = [{"label": l["label"], "text": l["raw_text"], "page": l["location"].get("page"), "line": l["location"]["line"],
                   "reasons": l["verify_reasons"] or (["not_usable"] if not l["usable"] else []), "quote": l["quote"]}
                  for l in res["labs"] if l["verify_reasons"] or not l["usable"]]
        return {
            "file_format": doc.kind, "document_type": res["document_profile"]["type"], "language": res["language_info"]["name"],
            "method": ("OCR" if any(p["method"] == "ocr" for p in pages) else "text extraction"),
            "page_count": doc.page_count, "pages": pages,
            "pages_processed_ok": sum(1 for p in pages if p["status"] == "ok"),
            "values_to_verify": verify,
        }

    def analyze_file(self, filename: str, data: bytes, content_type: str | None = None, **kw) -> list[dict[str, Any]]:
        """Validate + extract + analyse an uploaded file held in memory. Raises DocumentError."""
        doc = extract_document(filename, data, content_type,
                               ocr_engine=self._ocr_for_documents(), ocr_page=self._ocr_page_callback())
        return self.analyze_document(doc, **kw)

    def models_status(self) -> dict[str, Any]:
        """Which components are really available right now (no downloads are triggered)."""
        def has(mod: str) -> bool:
            import importlib.util
            try:
                return importlib.util.find_spec(mod) is not None
            except (ImportError, ValueError):
                return False

        clf = self.classifier
        return {
            "rule_engine": {"available": True, "version": __version__, "language": "en"},
            "context_classifier": ({"available": True, "name": getattr(clf, "name", "classifier"),
                                    "labels": list(getattr(clf, "labels", [])), "policy": self.classifier_policy,
                                    "threshold": self.classifier_threshold,
                                    "metrics": getattr(clf, "metrics", None)}
                                   if clf is not None else {"available": False}),
            "biogpt": {"available": bool(self.biogpt is not None and getattr(self.biogpt, "_backend", None) is not None),
                       "loaded": bool(self.biogpt is not None and getattr(self.biogpt, "_backend", None) is not None),
                       "torch_installed": has("torch"), "transformers_installed": has("transformers"),
                       "model": os.getenv("NUTRICHECK_BIOGPT_MODEL", "microsoft/biogpt")},
            "ocr": {"available": self._ocr is not None, "paddleocr_installed": has("paddleocr"),
                    "loaded": self._ocr is not None, "error": self._ocr_error},
            "documents": {"pymupdf": has("fitz"), "python_docx": has("docx"), "pandas": has("pandas")},
        }
