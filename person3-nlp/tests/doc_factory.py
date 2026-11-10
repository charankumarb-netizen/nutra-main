"""Builders for real test documents (PDF / DOCX / PNG) generated on the fly, no binary fixtures needed."""
from __future__ import annotations

import io


def make_pdf(pages: list[str], tables: dict[int, list[list[str]]] | None = None) -> bytes:
    """Digital PDF. `pages` = text of each page; `tables` = {page_index: rows} drawn as a ruled grid (detected by find_tables)."""
    import fitz

    doc = fitz.open()
    for i, text in enumerate(pages):
        page = doc.new_page()
        y = 60
        for line in text.split("\n"):
            page.insert_text((50, y), line, fontsize=11)
            y += 16
        if tables and i in tables:
            rows = tables[i]
            x0, y0, cw, rh = 50, y + 20, 110, 24
            ncol = max(len(r) for r in rows)
            for r in range(len(rows) + 1):
                page.draw_line((x0, y0 + r * rh), (x0 + ncol * cw, y0 + r * rh))
            for c in range(ncol + 1):
                page.draw_line((x0 + c * cw, y0), (x0 + c * cw, y0 + len(rows) * rh))
            for r, row in enumerate(rows):
                for c, cell in enumerate(row):
                    page.insert_text((x0 + c * cw + 4, y0 + r * rh + 16), cell, fontsize=10)
    data = doc.tobytes()
    doc.close()
    return data


def make_png_text(text: str = "scan", size=(600, 200)) -> bytes:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", size, "white")
    ImageDraw.Draw(img).text((10, 10), text, fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_scanned_pdf(n_pages: int = 1) -> bytes:
    """PDF whose pages are only images (no text layer) - what a scanner produces."""
    import fitz

    doc = fitz.open()
    for _ in range(n_pages):
        page = doc.new_page()
        page.insert_image(fitz.Rect(40, 40, 500, 300), stream=make_png_text("ferritin 8 ng/mL (image only)"))
    data = doc.tobytes()
    doc.close()
    return data


def make_encrypted_pdf() -> bytes:
    import fitz

    doc = fitz.open()
    doc.new_page().insert_text((50, 50), "secret")
    data = doc.tobytes(encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="pw")
    doc.close()
    return data


def make_docx(paragraphs: list[str] | None = None, tables: list[list[list[str]]] | None = None, merge_first_row: bool = False) -> bytes:
    import docx

    d = docx.Document()
    for p in paragraphs or []:
        d.add_paragraph(p)
    for rows in tables or []:
        t = d.add_table(rows=len(rows), cols=max(len(r) for r in rows))
        for i, row in enumerate(rows):
            for j, cell in enumerate(row):
                t.cell(i, j).text = cell
        if merge_first_row and len(rows[0]) > 1:
            t.cell(0, 0).merge(t.cell(0, len(rows[0]) - 1))
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


class FakeOCR:
    """Stands in for Person 1's MedicalOCREngine."""

    def __init__(self, text="Ferritin: 7 ng/mL\nVitamin D: 40 ng/mL", confidence=0.97, success=True):
        self.text, self.confidence, self.success, self.calls = text, confidence, success, 0

    def process_image(self, path, preprocess=True):
        self.calls += 1
        if not self.success:
            return {"success": False, "error": "boom"}
        return {"success": True, "text": self.text, "average_confidence": self.confidence,
                "line_count": self.text.count("\n") + 1, "image": str(path)}
