"""
Laboratory value extraction (English reports, OCR-tolerant, defensive).

Finds  <analyte name> ... <number> [unit]  patterns, converts units, guards against
doses / dates / reference ranges / OCR errors / invalid numbers, and classifies each
value against the reference table in lexicon.py.

Safety rules implemented here (each one has tests):
  * a negative number, scientific notation ("1e999") or a non-finite number is REJECTED, never read as a low value;
  * a number glued to letters that OCR often confuses with digits ("l8", "1O") is flagged, never guessed;
  * "12,5" (decimal comma) is flagged as ambiguous in English text;
  * a missing unit gives a PROVISIONAL value (the unit is assumed) that can never produce a definitive result;
  * an implausible value (calcium 85 mg/dL) is flagged for verification, never classified as normal;
  * a medication dose ("vitamin D 2000 IU/day") is not a measurement;
  * the laboratory's own reference range is read from the SAME line, so "45 (ref 15-150)" yields 45.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field

from .dates import DateHit, date_for_position, document_date, find_dates, temporal_cue
from .lexicon import (
    ANALYTES, DOSE_UNIT_REGEX, HEMOGLOBIN_THRESHOLD_BY_SEX, SUPPLEMENT_VERBS, UNIT_REGEX, Analyte,
)
from .multilingual import (
    MULTILINGUAL_ORDERED_TEST_CUES, MULTILINGUAL_REF_KEYS,
    MULTILINGUAL_SEPARATORS_PATTERN, parse_multilingual_number,
)
from .text_normalizer import normalize_unit

logger = logging.getLogger("nutricheck.lab_extractor")

_QUAL = r"(?:\s*,?\s*(?:serum|total|plasma|blood|level|levels|test|result|results|value|concentration|in\s+blood|in\s+serum|serico|serica|plasmatico|no\s+soro|nivel|dosagem))*"
_PAREN = r"(?:\s*\((?P<pin>[^)\d][^)]{0,35})\))?"
_SEP = MULTILINGUAL_SEPARATORS_PATTERN


_NUM = (
    r"(?P<cmp><=|>=|[<>]|≤|≥)?\s*(?P<sign>-(?=\d))?"
    r"(?:(?P<num>\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?(?!\d)|\d+(?:[.,]\d+)?(?!\d))"
    r"|(?P<ocr>(?=[0-9lioszb|.,]*\d)(?=[0-9lioszb|.,]*[lioszb|])[0-9lioszb|.,]{2,8}))"
)


_UNIT_AFTER = re.compile(r"\s*(?P<unit>" + UNIT_REGEX + r")")
_DOSE_AFTER = re.compile(DOSE_UNIT_REGEX)
_DATE_OR_RANGE_AFTER = re.compile(r"\s*(?:/\s*\d|-\s*\d|-\s*(?:oh|hydroxy)|\(?\s*oh\b|to\s+\d+(?:[.,]\d+)?\s*(?:ng|pg|ug|mg|g|%))")
_N = r"\d+(?:[.,]\d+)?"
_REF_KEY = MULTILINGUAL_REF_KEYS
_REF_AFTER = re.compile(
    r"[ \t]*(?:(?P<open>[(\[])[ \t]*)?(?P<key>" + _REF_KEY + r")?[ \t]*[:=]?[ \t]*"
    r"(?:(?P<lo>" + _N + r")[ \t]*(?:-|to|a|ate|bis)[ \t]*(?P<hi>" + _N + r")"
    r"|(?P<cmp><=|>=|[<>]|≤|≥)[ \t]*(?P<lim>" + _N + r"))"
    r"(?:[ \t]*(?P<unit>" + UNIT_REGEX + r"))?"
)
_SUPPLEMENT_BEFORE = re.compile(r"(?:" + SUPPLEMENT_VERBS + r")[^.\n]{0,30}$")
_IRON_CONTEXT_BEFORE = re.compile(r"(?:capacity|binding|saturation|transferrin|capacidade|saturacao)[^.\n]{0,30}$")
_DOSE_LIKE_UNITS = re.compile(r"^(?:(?:mg|ug|g)/(?:day|d|wee|wk|w|dos|h|hr|mon|dia)|iu/.*|mg/kg|ml/.*|units?/.*)$")
_OCR_DIGIT = {"l": "1", "i": "1", "|": "1", "o": "0", "s": "5", "z": "2", "b": "8"}


def parse_number(raw: str) -> float:
    """Parse a number in English or continental European (comma-decimal) format."""
    return parse_multilingual_number(raw)


@dataclass
class LabValue:
    analyte: str
    target: str
    label: str
    raw_text: str
    value: float
    comparator: str | None
    unit: str | None
    canonical_value: float | None
    canonical_unit: str
    status: str                  # deficient | borderline | normal | high | indeterminate
    assumed_unit: bool = False
    usable: bool = True
    weak: bool = False
    tier: int = 1
    start: int = 0
    end: int = 0
    warnings: list[str] = field(default_factory=list)
    lab_reference: dict | None = None     # the laboratory's own interval, e.g. {"low": 15, "high": 150, "raw": "15 - 150"}
    reference_check: str | None = None    # below | within | above  (value vs the lab's own interval)
    ref_conflict: bool = False            # lab interval and clinical cut-off disagree -> human review
    provisional: bool = False             # unit was assumed: may suggest, never decide
    verify_reasons: list[str] = field(default_factory=list)   # why a human must look at the original (codes)
    date: dict | None = None              # {"raw", "iso", "precision", "ambiguous", "candidates", "source"}
    temporal: str | None = None           # current | past | None
    ocr_candidates: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def classify(a: Analyte, v: float, comparator: str | None, deficient_below: float | None = None) -> str:
    """Classify a canonical value (optionally with a < / > comparator)."""
    thr = a.deficient_below if deficient_below is None else deficient_below
    normal_from = a.borderline_below if a.borderline_below is not None else thr
    cmp_ = comparator
    if cmp_ in (None, "="):
        if v < thr:
            return "deficient"
        if a.borderline_below is not None and v < a.borderline_below:
            return "borderline"
        if a.high_above is not None and v > a.high_above:
            return "high"
        return "normal"
    if cmp_ == "<":
        return "deficient" if v <= thr else "indeterminate"
    if cmp_ in ("<=", "≤"):
        return "deficient" if v < thr else "indeterminate"
    if cmp_ in (">", ">=", "≥"):
        if a.high_above is not None and v >= a.high_above:
            return "high" if cmp_ == ">" or v > a.high_above else "indeterminate"
        return "normal" if v >= normal_from else "indeterminate"
    return "indeterminate"


class LabExtractor:
    MAX_LABS = 300     # safety cap: a report cannot meaningfully contain more distinct values than this

    def __init__(self, sex: str | None = None, date_order: str | None = None) -> None:
        self.sex = sex
        self.date_order = date_order
        self._patterns: dict[str, re.Pattern] = {}
        self._before_excl: dict[str, re.Pattern] = {}
        for key, a in ANALYTES.items():
            pat = (
                r"(?<![a-z0-9])(?P<name>" + a.name_regex + r")(?![a-z0-9])"
                + a.qualifier_exclusions + _QUAL + _PAREN + _SEP + _NUM
            )
            self._patterns[key] = re.compile(pat)
            if a.before_exclusions:
                self._before_excl[key] = re.compile(a.before_exclusions)

    # ------------------------------------------------------------------
    def extract(self, norm: str, original: str | None = None) -> tuple[list[LabValue], list[dict]]:
        labs: list[LabValue] = []
        rejected: list[dict] = []
        seen: set[tuple] = set()
        original = original if original is not None else norm
        hits = find_dates(norm, self.date_order)
        doc_date = document_date(norm, hits)

        logger.info(f"[EXTRACTION TRACE] Original extracted text length: {len(original)}")
        logger.info(f"[EXTRACTION TRACE] Text snippet: {original[:200]!r}")

        for key, a in ANALYTES.items():
            for m in self._patterns[key].finditer(norm):
                if len(labs) >= self.MAX_LABS:
                    rejected.append({"analyte": key, "target": a.target, "raw_text": "(further values)", "reason": "too_many_values",
                                     "start": m.start(), "end": m.end()})
                    break
                raw_snip = original[m.start():m.end()].strip()
                if key in self._before_excl and self._before_excl[key].search(norm[max(0, m.start() - 28): m.start()]):
                    continue
                tok_start = m.start("num") if m.group("num") else m.start("ocr")
                tok_end = m.end("num") if m.group("num") else m.end("ocr")
                before = norm[max(0, m.start() - 45): m.start()]

                def reject(reason: str) -> None:
                    tail_tok = re.match(r"\S*", original[m.end():]).group(0)
                    rejected.append({"analyte": key, "target": a.target, "raw_text": (raw_snip + tail_tok)[:60], "reason": reason,
                                     "start": m.start(), "end": m.end()})

                verify: list[str] = []
                warnings: list[str] = []
                ocr_cands: list[float] = []
                after = norm[tok_end: tok_end + 24]
                num_text = m.group("num") or m.group("ocr")

                # --- numbers glued to letters / OCR-confusable characters ---------------------------
                if re.match(r"e[+-]?\d", after):
                    reject("scientific_notation_or_non_finite")
                    continue
                glued = after[:1].isalpha() and not re.match(r"[a-zµμ]{1,6}\s*/", after) and not after.startswith("fl")
                if m.group("ocr") or glued:
                    token = re.match(r"[0-9a-z|.,]+", norm[tok_start:tok_start + 10]).group(0)
                    if not (re.fullmatch(r"[0-9lioszb|.,]+", token) and re.search(r"\d", token)):
                        reject("unreadable_number")
                        continue
                    tok_end = tok_start + len(token)
                    after = norm[tok_end: tok_end + 24]
                    guess = "".join(_OCR_DIGIT.get(ch, ch) for ch in token)
                    try:
                        value = parse_number(guess)
                    except ValueError:
                        reject("unreadable_number")
                        continue
                    ocr_cands = [value]
                    num_text = token
                    verify.append("ambiguous_ocr_characters")
                    warnings.append(
                        f"The number '{original[tok_start:tok_end]}' contains characters that OCR often confuses with digits "
                        f"(l/I/1, O/0, S/5, B/8): it could be {value:g}, but this was not assumed. Check the original image.")
                else:
                    try:
                        value = parse_number(num_text)
                    except ValueError:
                        reject("unparseable_number")
                        continue

                # --- structural rejections -----------------------------------------------------------
                if m.group("sign"):
                    reject("negative_value")
                    continue
                if re.match(r"e[+-]?\d", after):
                    reject("scientific_notation_or_non_finite")
                    continue
                if not math.isfinite(value):
                    reject("scientific_notation_or_non_finite")
                    continue
                if _DATE_OR_RANGE_AFTER.match(after):
                    reject("date_or_reference_range")
                    continue
                if _DOSE_AFTER.match(after):
                    continue  # medication dose, silently ignored
                if key == "serum_iron" and _IRON_CONTEXT_BEFORE.search(norm[max(0, m.start() - 40): m.end()]):
                    continue
                unit_after = after
                dm = re.match(r",(\d{1,2})(?!\d)", after)
                if dm and m.group("num") and "," not in m.group("num"):
                    # Support trailing decimal comma if separated by tokenizer
                    try:
                        value = parse_number(f"{num_text}.{dm.group(1)}")
                        num_text = f"{num_text},{dm.group(1)}"
                    except ValueError:
                        pass
                    unit_after = after[dm.end():]
                    tok_end += dm.end()

                unit_m = _UNIT_AFTER.match(unit_after)
                unit_raw = unit_m.group("unit") if unit_m else None
                unit = normalize_unit(unit_raw) if unit_raw else None
                if unit is None and m.group("pin"):
                    cand = normalize_unit(m.group("pin").strip())
                    if cand and re.fullmatch(UNIT_REGEX, cand):
                        unit = cand
                if unit is not None and _DOSE_LIKE_UNITS.match(unit):
                    continue
                if unit is None and _SUPPLEMENT_BEFORE.search(before):
                    continue  # "taking vitamin D 2000"

                d_hit, d_src = date_for_position(norm, hits, m.start(), tok_end, doc_date)
                date = None
                if d_hit is not None:
                    date = {**d_hit.to_dict(), "source": d_src}
                    if d_hit.ambiguous:
                        warnings.append(f"The date '{d_hit.raw}' is ambiguous (day/month order unknown); it was not guessed.")
                        verify.append("ambiguous_date")
                temporal = temporal_cue(norm, m.start(), tok_end)

                dedupe_key = (key, num_text, m.group("cmp"), unit, (date or {}).get("iso"), (date or {}).get("raw"))
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)

                tail_from = tok_end + (unit_m.end() if unit_m else 0)
                line_end = norm.find("\n", tail_from)
                tail = norm[tail_from: (line_end if line_end != -1 else len(norm))][:90]
                lab = self._build(a, key, m, raw_snip, value, unit,
                                  self._parse_reference(tail, a, unit, original[tail_from: tail_from + len(tail)]),
                                  verify, warnings, ocr_cands, date, temporal)
                if lab is None:
                    reject(f"unit_mismatch:{unit}")
                    continue

                matched_name = m.group("name")
                logger.info(f"[EXTRACTION TRACE] Recognized biomarker: '{matched_name}' -> analyte '{key}' (target: '{a.target}')")
                logger.info(f"[EXTRACTION TRACE] Matched numerical value string: '{num_text}'")
                logger.info(f"[EXTRACTION TRACE] Detected unit: '{unit}' (raw: '{unit_raw}')")
                logger.info(f"[EXTRACTION TRACE] Normalized numerical value: {value:g} (canonical: {lab.canonical_value:g} {lab.canonical_unit})")
                labs.append(lab)

        labs.sort(key=lambda l: l.start)
        logger.info(f"[EXTRACTION TRACE] Final extraction results: {len(labs)} lab(s) extracted:")
        for l in labs:
            logger.info(f"   -> {l.label} ({l.analyte}): {l.value:g} {l.unit} [status={l.status}, usable={l.usable}]")
        return labs, rejected

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_reference(tail: str, a: Analyte, value_unit: str | None, orig_tail: str | None = None) -> dict | None:
        """Read "(Ref: 15 - 150)", "Reference range: 30 to 400 ng/mL", "15 - 150" or "(> 20)" right after a value."""
        mm = _REF_AFTER.match(tail)
        if not mm:
            return None
        raw_ref = (orig_tail if orig_tail is not None and len(orig_tail) == len(tail) else tail)[mm.start():mm.end()].strip(" ()[]")
        has_marker = bool(mm.group("open") or mm.group("key"))
        if mm.group("lo") is None and not has_marker:
            return None                     # a bare "> 20" is too ambiguous to be a reference
        unit = normalize_unit(mm.group("unit")) if mm.group("unit") else (value_unit or a.default_unit)
        if unit not in a.units:
            return None                     # reference in a unit we cannot convert -> ignore it
        factor = a.units[unit]
        anchor = a.borderline_below or a.deficient_below      # the clinical cut-off this analyte is judged against

        def plausible_low(v: float) -> bool:
            # A real *lower* reference limit sits near the clinical cut-off (typically 0.3x - 1.5x of it).
            # A range printed next to ANOTHER analyte's value is ignored rather than allowed to reclassify a value.
            if not (0.3 * anchor <= v <= 1.5 * anchor):
                return False
            return a.high_above is None or v < a.high_above

        try:
            if mm.group("lo") is not None:
                lo, hi = float(mm.group("lo")) * factor, float(mm.group("hi")) * factor
                if lo >= hi or not plausible_low(lo) or hi > a.plausible[1]:
                    return None
                return {"low": round(lo, 4), "high": round(hi, 4), "unit": a.canonical_unit, "raw": raw_ref}
            lim = float(mm.group("lim")) * factor
        except ValueError:
            return None
        cmp_ = mm.group("cmp")
        if cmp_ in (">", ">=", "≥"):
            if not plausible_low(lim):
                return None
            return {"low": round(lim, 4), "high": None, "unit": a.canonical_unit, "raw": raw_ref}
        return {"low": None, "high": round(lim, 4), "unit": a.canonical_unit, "raw": raw_ref}

    def _build(self, a: Analyte, key: str, m: re.Match, raw: str, value: float, unit: str | None,
               ref: dict | None, verify: list[str], warnings: list[str], ocr_cands: list[float],
               date: dict | None, temporal: str | None) -> LabValue | None:
        assumed = False
        cmp_ = m.group("cmp")

        # total calcium reported in mmol/L with an ionized-looking value
        if key == "calcium_total" and unit == "mmol/l" and 0.5 <= value <= 1.5:
            a = ANALYTES["calcium_ionized"]
            key = "calcium_ionized"
            warnings.append("Calcium in mmol/L with a value typical of ionized calcium: interpreted as ionized calcium.")

        if unit is None:
            unit = a.default_unit
            assumed = True
            verify.append("unit_missing")
            warnings.append(f"Unit missing: {a.default_unit} was ASSUMED. The result is provisional and needs confirmation.")
        if unit not in a.units:
            return None
        factor = a.units[unit]
        canon = value * factor
        lo, hi = a.plausible

        deficient_below = None
        if key == "hemoglobin":
            deficient_below = HEMOGLOBIN_THRESHOLD_BY_SEX.get(self.sex, 12.0)
            if self.sex is None:
                warnings.append("Sex not stated: the 12 g/dL anemia threshold was used.")

        usable = True
        if any(v in verify for v in ("ambiguous_ocr_characters", "ambiguous_decimal_format")):
            status, usable = "indeterminate", False
        elif not (lo <= canon <= hi):
            status, usable = "indeterminate", False
            verify.append("implausible_value")
            warnings.append(f"Implausible value ({value:g} {unit}): possible OCR, typing or unit error. Verify against the original; it was not classified.")
        else:
            status = classify(a, canon, cmp_, deficient_below)
            if status == "indeterminate":
                usable = False
                verify.append("comparator_not_decisive")
                warnings.append("The comparator (<, >) does not allow a safe classification.")

        ref_check = None
        ref_conflict = False
        if ref is not None and usable and cmp_ in (None, "="):
            lo_, hi_ = ref.get("low"), ref.get("high")
            if lo_ is not None and canon < lo_:
                ref_check = "below"
            elif hi_ is not None and canon > hi_:
                ref_check = "above"
            else:
                ref_check = "within"
            shown = ref["raw"]
            if status == "normal" and ref_check == "below":
                # Never change the status silently: the clinical cut-off stays in charge, a human checks the difference.
                ref_conflict = True
                warnings.append(f"Below the laboratory's own reference range ({shown}) but above the clinical cut-off used here: please review.")
            elif status == "deficient" and ref_check in ("within", "above"):
                ref_conflict = True
                warnings.append(f"Within the laboratory's own reference range ({shown}) but below the clinical cut-off used here: please review.")

        return LabValue(
            analyte=key, target=a.target, label=a.label, raw_text=raw, value=value,
            comparator=cmp_, unit=unit, canonical_value=round(canon, 4), canonical_unit=a.canonical_unit,
            status=status, assumed_unit=assumed, usable=usable, weak=a.weak, tier=a.tier,
            start=m.start(), end=m.end(), warnings=warnings,
            lab_reference=ref, reference_check=ref_check, ref_conflict=ref_conflict,
            provisional=assumed, verify_reasons=list(dict.fromkeys(verify)), date=date, temporal=temporal,
            ocr_candidates=ocr_cands,
        )
