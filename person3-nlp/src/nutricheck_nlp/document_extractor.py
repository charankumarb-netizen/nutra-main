"""
Document extraction layer  (Input Layer -> Document Processing).

One function per supported format, all returning the same standardised object:

    extract_pdf(data)    digital PDFs (PyMuPDF), tables kept as "Analyte: value unit" lines,
                         scanned pages sent to the OCR callback (Person 1's PaddleOCR) when available
    extract_docx(data)   Word files (python-docx), paragraphs AND tables in document order
    extract_csv(data)    lab tables (long or wide format) and free-text columns (pandas)
    extract_txt(data)    plain text with encoding detection
    extract_image(path)  photos/scans through Person 1's OCR engine

`extract_document(filename, data, content_type, ocr_engine)` validates the upload first
(extension, MIME type, magic bytes, size) and dispatches to the right function.

Nothing is written to disk here: everything is processed from bytes in memory. (Images are the
single exception because PaddleOCR reads from a path - the API deletes that temp file in a
`finally` block.)

Every function returns an `ExtractedDocument` whose `pages` list lets the pipeline translate a
character offset in `text` back to a page and line number, so each finding can point at its source.
"""

from __future__ import annotations

import io
import os
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .lexicon import ANALYTES
from .text_normalizer import MAX_TEXT_CHARS, normalize_text

# ---------------------------------------------------------------------------
# Limits / allowed types (override with environment variables)
# ---------------------------------------------------------------------------

MAX_FILE_BYTES = int(float(os.getenv("NUTRICHECK_MAX_UPLOAD_MB", "10")) * 1024 * 1024)
MAX_PDF_PAGES = int(os.getenv("NUTRICHECK_MAX_PDF_PAGES", "60"))
MAX_CSV_ROWS = int(os.getenv("NUTRICHECK_MAX_CSV_ROWS", "5000"))
SCANNED_PAGE_MIN_CHARS = 25          # fewer extractable characters than this -> treat as scanned
OCR_RENDER_DPI = 200

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ALLOWED_EXT = {".pdf", ".docx", ".txt", ".csv"} | IMAGE_EXT

_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_GENERIC_MIME = {"application/octet-stream", "binary/octet-stream", ""}
ALLOWED_MIME: dict[str, set[str]] = {
    ".pdf": {"application/pdf", "application/x-pdf"},
    ".docx": {_DOCX_MIME},
    ".txt": {"text/plain", "text/markdown"},
    ".csv": {"text/csv", "application/csv", "text/plain", "application/vnd.ms-excel"},
    ".jpg": {"image/jpeg"}, ".jpeg": {"image/jpeg"}, ".png": {"image/png"}, ".bmp": {"image/bmp", "image/x-ms-bmp"},
    ".tif": {"image/tiff"}, ".tiff": {"image/tiff"}, ".webp": {"image/webp"},
}

OcrPage = Callable[[bytes], "dict[str, Any] | None"]   # png bytes -> {"text":..., "average_confidence":...}


class DocumentError(Exception):
    """A problem with the uploaded document. `status` is the matching HTTP status code."""

    def __init__(self, message: str, status: int = 422) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass
class Record:
    """One patient/note inside a document (a CSV can hold many; other formats hold one)."""

    record_id: str
    text: str
    pages: list[dict] = field(default_factory=list)   # [{"page":1,"start":0,"end":120,"method":"text"}]


@dataclass
class ExtractedDocument:
    filename: str
    kind: str                               # pdf | docx | txt | csv | image
    records: list[Record]
    tables: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ocr_confidence: float | None = None
    needs_review: bool = False              # e.g. a scanned page could not be read
    page_count: int = 1
    page_info: list[dict] = field(default_factory=list)   # per page: method, characters, ocr_confidence, status, notes
    ocr_lines: list[dict] | None = None                   # per-line OCR confidences (images)

    @property
    def text(self) -> str:
        return "\n".join(r.text for r in self.records)

    def meta(self) -> dict:
        return {
            "filename": self.filename, "kind": self.kind, "page_count": self.page_count,
            "records": len(self.records), "tables": self.tables, "warnings": self.warnings,
            "ocr_confidence": self.ocr_confidence, "needs_review": self.needs_review,
        }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _looks_like(data: bytes, ext: str) -> bool:
    head = data[:16]
    if ext == ".pdf":
        return b"%PDF" in data[:1024]
    if ext == ".docx":
        if not data.startswith(b"PK\x03\x04"):
            return False
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                return "word/document.xml" in z.namelist()
        except zipfile.BadZipFile:
            return False
    if ext in (".jpg", ".jpeg"):
        return head.startswith(b"\xff\xd8\xff")
    if ext == ".png":
        return head.startswith(b"\x89PNG\r\n\x1a\n")
    if ext == ".bmp":
        return head.startswith(b"BM")
    if ext in (".tif", ".tiff"):
        return head[:4] in (b"II*\x00", b"MM\x00*")
    if ext == ".webp":
        return head.startswith(b"RIFF") and head[8:12] == b"WEBP"
    if ext in (".txt", ".csv"):
        return b"\x00" not in data[:4096]          # binary files are not text
    return False


def validate_upload(filename: str, data: bytes, content_type: str | None = None) -> str:
    """Validate extension, MIME type, size and magic bytes. Returns the lower-case extension."""
    ext = Path(filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise DocumentError(f"Unsupported file type '{ext or '(no extension)'}'. "
                            f"Permitidos: {sorted(ALLOWED_EXT)}", 415)
    if not data:
        raise DocumentError("Arquivo vazio.", 400)
    if len(data) > MAX_FILE_BYTES:
        raise DocumentError(f"File too large (maximum {MAX_FILE_BYTES // (1024 * 1024)} MB).", 413)
    ctype = (content_type or "").split(";")[0].strip().lower()
    if ctype not in _GENERIC_MIME and ctype not in ALLOWED_MIME[ext]:
        raise DocumentError(f"The MIME type '{ctype}' does not match the extension '{ext}'.", 415)
    if not _looks_like(data, ext):
        raise DocumentError(f"The file content does not match the format '{ext}' (corrupted or renamed file).", 415)
    return ext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\u00a0]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip("\n")


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="replace")  # pragma: no cover - latin-1 never fails


def _join_pages(page_texts: list[tuple[int | None, str, str]]) -> tuple[str, list[dict], list[str]]:
    """[(page_no, text, method)] -> (full text, page spans, warnings). Truncates at MAX_TEXT_CHARS."""
    parts: list[str] = []
    spans: list[dict] = []
    warnings: list[str] = []
    pos = 0
    for page_no, text, method in page_texts:
        text = _clean(text)
        if parts:
            parts.append("\n")
            pos += 1
        if pos + len(text) > MAX_TEXT_CHARS:
            keep = max(0, MAX_TEXT_CHARS - pos)
            text = text[:keep]
            warnings.append(f"Text truncated at {MAX_TEXT_CHARS} characters (page {page_no}).")
        parts.append(text)
        spans.append({"page": page_no, "start": pos, "end": pos + len(text), "method": method})
        pos += len(text)
        if pos >= MAX_TEXT_CHARS:
            break
    return "".join(parts), spans, warnings


def _norm_header(h: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", normalize_text(str(h))).strip()


def table_to_lines(rows: list[list[str]]) -> list[str]:
    """
    Convert a table into "Analyte: value unit (Ref: ...)" lines.

    Recognises the usual lab layout  Test | Result | Unit | Reference  (the header may be preceded
    by a title row such as a merged "Test results" cell). Other tables with a header row become "header: cell"
    pairs, which the lab extractor still reads ("Ferritin (ng/mL): 8"); headerless tables become one line per row.
    """
    rows = [[(c or "").strip() for c in r] for r in rows if any((c or "").strip() for c in r)]
    if not rows:
        return []
    lines: list[str] = []
    for hi in range(min(3, len(rows))):                       # the real header may sit under a title row
        roles = _roles([_norm_header(c) for c in rows[hi]])
        if roles["analyte"] is None or roles["value"] is None:
            continue
        for r in rows[hi + 1:]:
            name = r[roles["analyte"]] if roles["analyte"] < len(r) else ""
            val = r[roles["value"]] if roles["value"] < len(r) else ""
            unit = r[roles["unit"]] if roles["unit"] is not None and roles["unit"] < len(r) else ""
            ref = r[roles["ref"]] if roles["ref"] is not None and roles["ref"] < len(r) else ""
            if not name or not val:
                continue
            line = f"{name}: {val}"
            if unit and unit.lower() not in val.lower():
                line += f" {unit}"
            if ref:
                line += f" (Ref: {ref})"
            lines.append(line)
        return lines
    header = [_norm_header(c) for c in rows[0]]
    if _looks_like_header(header) and len(rows) > 1:
        for r in rows[1:]:
            for h, c in zip(rows[0], r):
                if c and h:
                    lines.append(f"{h}: {c}")
        return lines
    return [" ".join(c for c in r if c) for r in rows]        # headerless: "cell cell cell" per row


_ANALYTE_COLS = {"analyte", "test", "name", "exam", "item", "parameter", "investigation", "test name", "analyte name"}
_VALUE_COLS = {"result", "results", "value", "res", "observed value", "test result"}
_UNIT_COLS = {"unit", "units"}
_REF_COLS = {"reference", "reference range", "reference interval", "reference value", "reference values", "ref", "ref range", "normal range", "range", "interval"}
_TEXT_COLS = {"text", "note", "notes", "transcription", "clinical notes", "clinical note", "comment", "comments", "description",
              "report", "history", "evolution", "findings", "impression", "summary"}
_ID_COLS = {"id", "patient", "patient id", "patient_id", "patientid", "mrn", "record", "record id", "subject", "subject id"}


def _roles(header: list[str]) -> dict[str, int | None]:
    def find(options: set[str]) -> int | None:
        for i, h in enumerate(header):
            if h in options:
                return i
        return None

    return {"analyte": find(_ANALYTE_COLS), "value": find(_VALUE_COLS), "unit": find(_UNIT_COLS), "ref": find(_REF_COLS)}


def _looks_like_header(header: list[str]) -> bool:
    """A header row has labels, not values: no cell starts with a digit ("18 ng/mL", "8")."""
    cells = [h for h in header if h]
    return bool(cells) and not any(re.match(r"\d", h) for h in cells)


# ---------------------------------------------------------------------------
# TXT
# ---------------------------------------------------------------------------


def extract_txt(data: bytes, filename: str = "text.txt") -> ExtractedDocument:
    text, spans, warn = _join_pages([(None, _decode(data), "text")])
    if not text.strip():
        warn.append("The text file is empty.")
    return ExtractedDocument(filename, "txt", [Record("1", text, spans)], warnings=warn)


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


def _page_text_with_tables(page) -> tuple[str, list[dict]]:
    """Page text where detected tables are replaced by structured "Analyte: value unit" lines."""
    tables_meta: list[dict] = []
    try:
        found = page.find_tables()
        tables = list(found.tables) if found is not None else []
    except Exception:  # find_tables is heuristic/optional
        tables = []
    if not tables:
        return page.get_text("text"), tables_meta

    bboxes = [t.bbox for t in tables]
    words = page.get_text("words")          # (x0, y0, x1, y1, word, block, line, word_no)
    keep: dict[tuple[int, int], list[tuple[float, str]]] = {}
    for x0, y0, x1, y1, w, b, ln, _wn in words:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if any(bx0 <= cx <= bx1 and by0 <= cy <= by1 for bx0, by0, bx1, by1 in bboxes):
            continue
        keep.setdefault((b, ln), []).append((x0, w))
    outside = [" ".join(w for _, w in sorted(v)) for _, v in sorted(keep.items())]
    table_lines: list[str] = []
    import fitz

    for t in tables:
        try:   # cell text read through the normal text extractor keeps the spaces (Table.extract() can drop them)
            rows = [[" ".join(page.get_text("text", clip=fitz.Rect(c)).split()) if c else "" for c in row.cells] for row in t.rows]
        except Exception:
            rows = [[c or "" for c in r] for r in t.extract()]
        lines = table_to_lines(rows)
        tables_meta.append({"rows": len(rows), "columns": len(rows[0]) if rows else 0, "lines": len(lines)})
        table_lines.extend(lines)
    return "\n".join(outside + table_lines), tables_meta


def extract_pdf(data: bytes, filename: str = "document.pdf", ocr_page: OcrPage | None = None) -> ExtractedDocument:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise DocumentError("PyMuPDF is not installed (pip install pymupdf).", 503) from exc
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise DocumentError(f"Invalid or corrupted PDF: {exc}", 422) from exc
    try:
        if doc.needs_pass:
            raise DocumentError("The PDF is password-protected and cannot be read.", 422)
        if doc.page_count == 0:
            raise DocumentError("The PDF has no pages.", 422)
        warnings: list[str] = []
        total = doc.page_count
        if total > MAX_PDF_PAGES:
            warnings.append(f"The PDF has {total} pages; only the first {MAX_PDF_PAGES} were processed.")
        pages: list[tuple[int, str, str]] = []
        tables_meta: list[dict] = []
        confidences: list[float] = []
        page_info: list[dict] = []
        needs_review = False
        for i in range(min(total, MAX_PDF_PAGES)):
            n_warn_before = len(warnings)
            page_conf = None
            page = doc[i]
            try:
                text, tmeta = _page_text_with_tables(page)
            except Exception as exc:  # pragma: no cover - defensive
                text, tmeta = "", []
                warnings.append(f"Page {i + 1}: text extraction failed ({exc}).")
            for t in tmeta:
                tables_meta.append({"page": i + 1, **t})
            method = "table" if tmeta else "text"
            has_images = bool(page.get_images())
            if len(text.strip()) < SCANNED_PAGE_MIN_CHARS and has_images:
                # (almost) no text layer but the page holds images -> a scan: send it to the OCR (Person 1)
                if ocr_page is None:
                    warnings.append(f"Page {i + 1}: looks scanned, but OCR is not available.")
                    needs_review = True
                else:
                    try:
                        png = page.get_pixmap(dpi=OCR_RENDER_DPI).tobytes("png")
                        res = ocr_page(png)
                    except Exception as exc:
                        res = None
                        warnings.append(f"Page {i + 1}: OCR failed ({exc}).")
                    if res and str(res.get("success", True)) != "False" and str(res.get("text", "")).strip():
                        text = (text.strip() + "\n" + str(res["text"])).strip()      # keep any digital text too
                        method = "ocr"
                        if res.get("average_confidence") is not None:
                            page_conf = float(res["average_confidence"])
                            confidences.append(page_conf)
                        warnings.append(f"Page {i + 1}: scanned page read by OCR.")
                    else:
                        warnings.append(f"Page {i + 1}: no readable text (OCR returned no text).")
                        needs_review = True
            elif not text.strip():
                warnings.append(f"Page {i + 1}: page has no text.")
            pages.append((i + 1, text, method))
            p_status = "ok"
            if not text.strip():
                p_status = "failed"
            elif method == "ocr" and (page_conf is None or page_conf < 0.85):
                p_status = "needs_verification"
            page_info.append({"page": i + 1, "method": method, "characters": len(text), "ocr_confidence": page_conf,
                              "status": p_status, "notes": warnings[n_warn_before:]})
        full, spans, w2 = _join_pages(pages)
        warnings += w2
        if not full.strip():
            needs_review = True
            warnings.append("No text could be extracted from the PDF.")
        conf = (sum(confidences) / len(confidences)) if confidences else None
        return ExtractedDocument(filename, "pdf", [Record("1", full, spans)], tables_meta, warnings, conf,
                                 needs_review, page_count=total, page_info=page_info)
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------


def extract_docx(data: bytes, filename: str = "document.docx") -> ExtractedDocument:
    try:
        import docx
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except ImportError as exc:  # pragma: no cover
        raise DocumentError("python-docx is not installed (pip install python-docx).", 503) from exc
    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:
        raise DocumentError(f"Invalid or corrupted DOCX: {exc}", 422) from exc

    lines: list[str] = []
    tables_meta: list[dict] = []
    body = document.element.body
    for child in body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            txt = Paragraph(child, document).text
            if txt.strip():
                lines.append(txt)
        elif tag == "tbl":
            table = Table(child, document)
            rows: list[list[str]] = []
            for row in table.rows:
                cells: list[str] = []
                prev = None
                for cell in row.cells:           # merged cells repeat: keep one copy
                    if cell._tc is prev:
                        continue
                    prev = cell._tc
                    cells.append(cell.text.strip().replace("\n", " "))
                rows.append(cells)
            tl = table_to_lines(rows)
            tables_meta.append({"page": None, "rows": len(rows), "columns": max((len(r) for r in rows), default=0), "lines": len(tl)})
            lines.extend(tl)
    full, spans, warn = _join_pages([(None, "\n".join(lines), "text")])
    if not full.strip():
        warn.append("The Word document contains no text.")
    return ExtractedDocument(filename, "docx", [Record("1", full, spans)], tables_meta, warn,
                             needs_review=not full.strip())


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def extract_csv(data: bytes, filename: str = "data.csv") -> ExtractedDocument:
    import pandas as pd

    raw = _decode(data)
    try:
        df = pd.read_csv(io.StringIO(raw), sep=None, engine="python", dtype=str, keep_default_na=False,
                         nrows=MAX_CSV_ROWS + 1, skipinitialspace=True)
    except Exception as exc:
        raise DocumentError(f"Invalid CSV: {exc}", 422) from exc
    if df.empty or len(df.columns) == 0:
        raise DocumentError("The CSV has no data rows.", 422)
    warnings: list[str] = []
    if len(df) > MAX_CSV_ROWS:
        df = df.iloc[:MAX_CSV_ROWS]
        warnings.append(f"CSV has more than {MAX_CSV_ROWS} rows: only the first rows were processed.")

    cols = list(df.columns)
    header = [_norm_header(c) for c in cols]
    roles = _roles(header)
    id_idx = next((i for i, h in enumerate(header) if h in _ID_COLS), None)
    text_idx = [i for i, h in enumerate(header) if h in _TEXT_COLS]
    long_format = roles["analyte"] is not None and roles["value"] is not None

    groups: dict[str, list[str]] = {}
    order: list[str] = []

    def add(rec_id: str, line: str) -> None:
        if rec_id not in groups:
            groups[rec_id] = []
            order.append(rec_id)
        if line.strip():
            groups[rec_id].append(line)

    for n, (_, row) in enumerate(df.iterrows(), start=1):
        cells = [str(v).strip() for v in row.tolist()]
        if id_idx is not None and cells[id_idx]:
            rec_id = cells[id_idx]                      # explicit patient / record id column
        elif long_format and not text_idx:
            rec_id = "1"                                # one lab table = one patient
        else:
            rec_id = str(n)                             # one free-text note / wide row = one record
        if text_idx:
            for i in text_idx:
                add(rec_id, cells[i])
        if long_format:
            for line in table_to_lines([[str(c) for c in cols], cells]):
                add(rec_id, line)
        if not text_idx and not long_format:
            for i, (h, c) in enumerate(zip(cols, cells)):
                if i == id_idx or not c:
                    continue
                add(rec_id, f"{h}: {c}")

    records: list[Record] = []
    for rid in order:
        full, spans, w = _join_pages([(None, "\n".join(groups[rid]), "table")])
        warnings += w
        records.append(Record(str(rid), full, spans))
    if not any(r.text.strip() for r in records):
        raise DocumentError("The CSV has no usable text.", 422)
    layout = "long format (test/result)" if long_format else ("text columns" if text_idx else "wide format")
    meta = [{"page": None, "rows": len(df), "columns": len(cols), "lines": sum(len(g) for g in groups.values()), "layout": layout}]
    return ExtractedDocument(filename, "csv", records, meta, warnings)


# ---------------------------------------------------------------------------
# Image (Person 1 OCR)
# ---------------------------------------------------------------------------


def extract_image(path: str | Path, ocr_engine: Any, filename: str | None = None) -> ExtractedDocument:
    """OCR an image through Person 1's MedicalOCREngine (or any object with process_image(path))."""
    if ocr_engine is None:
        raise DocumentError("OCR is unavailable: install requirements-ocr.txt (PaddleOCR).", 503)
    res = ocr_engine.process_image(str(path))
    return image_result_to_document(res, filename or Path(str(path)).name)


def image_result_to_document(res: dict, filename: str) -> ExtractedDocument:
    if not isinstance(res, dict) or not res.get("success", False):
        err = res.get("error", "unknown failure") if isinstance(res, dict) else "invalid result"
        raise DocumentError(f"OCR could not read the image: {err}", 422)
    full, spans, warn = _join_pages([(1, str(res.get("text", "")), "ocr")])
    conf = res.get("average_confidence")
    review = not full.strip() or (conf is not None and conf < 0.85)
    if not full.strip():
        warn.append("OCR found no text in the image.")
    c = float(conf) if conf is not None else None
    info = [{"page": 1, "method": "ocr", "characters": len(full), "ocr_confidence": c,
             "status": "failed" if not full.strip() else ("needs_verification" if review else "ok"), "notes": list(warn)}]
    return ExtractedDocument(filename, "image", [Record("1", full, spans)], [], warn, c, review,
                             page_info=info, ocr_lines=res.get("lines") if isinstance(res.get("lines"), list) else None)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def extract_document(filename: str, data: bytes, content_type: str | None = None,
                     ocr_engine: Any = None, ocr_page: OcrPage | None = None) -> ExtractedDocument:
    """
    Validate and extract any supported upload from memory.

    `ocr_engine` (images) must expose process_image(path) -> dict like Person 1's engine.
    `ocr_page` (scanned PDF pages) is a callback png-bytes -> {"text", "average_confidence"}.
    """
    ext = validate_upload(filename, data, content_type)
    if ext == ".pdf":
        return extract_pdf(data, filename, ocr_page)
    if ext == ".docx":
        return extract_docx(data, filename)
    if ext == ".csv":
        return extract_csv(data, filename)
    if ext == ".txt":
        return extract_txt(data, filename)
    # images: PaddleOCR needs a path -> short-lived temp file, always removed
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        return extract_image(tmp_path, ocr_engine, filename)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def known_analyte_header(header: str) -> bool:
    """True if a CSV column header names a known analyte (used by tests / wide-format hints)."""
    h = normalize_text(header)
    return any(re.search(a.name_regex, h) for a in ANALYTES.values())
