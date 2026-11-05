"""Render the Stage 3 technical Markdown report to a Unicode-safe PDF."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, HRFlowable, Image, KeepTogether, PageBreak,
    PageTemplate, Paragraph, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
NAVY = colors.HexColor("#12364A")
TEAL = colors.HexColor("#087B70")
MINT = colors.HexColor("#DDF2EC")
INK = colors.HexColor("#132A38")
MUTED = colors.HexColor("#5E7380")
LINE = colors.HexColor("#D8E4E5")
BG = colors.HexColor("#F4F7F6")
AMBER = colors.HexColor("#E8A727")


def register_fonts() -> tuple[str, str]:
    regular_candidates = [Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/aptos.ttf")]
    bold_candidates = [Path("C:/Windows/Fonts/arialbd.ttf"), Path("C:/Windows/Fonts/aptos-bold.ttf")]
    regular = next((p for p in regular_candidates if p.exists()), None)
    bold = next((p for p in bold_candidates if p.exists()), None)
    if not regular or not bold:
        raise FileNotFoundError("A Unicode TrueType font was not found in C:/Windows/Fonts")
    pdfmetrics.registerFont(TTFont("R2AI", str(regular)))
    pdfmetrics.registerFont(TTFont("R2AI-Bold", str(bold)))
    pdfmetrics.registerFontFamily("R2AI", normal="R2AI", bold="R2AI-Bold")
    return "R2AI", "R2AI-Bold"


def escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def inline_markup(text: str) -> str:
    value = escape(text.strip())
    value = re.sub(r"`([^`]+)`", r'<font name="R2AI-Bold" color="#087B70">\1</font>', value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", value)
    return value


def styles():
    base = getSampleStyleSheet()
    result = {
        "body": ParagraphStyle("body", parent=base["BodyText"], fontName="R2AI", fontSize=9.3,
                               leading=14, textColor=INK, spaceAfter=5),
        "bullet": ParagraphStyle("bullet", parent=base["BodyText"], fontName="R2AI", fontSize=9.1,
                                 leading=13.5, textColor=INK, leftIndent=14, firstLineIndent=-8, spaceAfter=3),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontName="R2AI-Bold", fontSize=20,
                             leading=24, textColor=NAVY, spaceBefore=14, spaceAfter=8, keepWithNext=True),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName="R2AI-Bold", fontSize=14,
                             leading=18, textColor=TEAL, spaceBefore=11, spaceAfter=6, keepWithNext=True),
        "h3": ParagraphStyle("h3", parent=base["Heading3"], fontName="R2AI-Bold", fontSize=11,
                             leading=14, textColor=NAVY, spaceBefore=8, spaceAfter=4, keepWithNext=True),
        "code": ParagraphStyle("code", parent=base["Code"], fontName="R2AI", fontSize=7.8,
                               leading=11, textColor=NAVY, backColor=BG, borderColor=LINE,
                               borderWidth=.5, borderPadding=7, spaceBefore=4, spaceAfter=7),
        "small": ParagraphStyle("small", parent=base["BodyText"], fontName="R2AI", fontSize=7.5,
                                leading=10, textColor=MUTED),
    }
    return result


class NumberedDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, **kwargs):
        super().__init__(filename, **kwargs)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="main")
        self.addPageTemplates(PageTemplate(id="content", frames=frame, onPage=self.draw_page))

    def draw_page(self, canvas, doc):
        canvas.saveState()
        canvas.setFillColor(NAVY)
        canvas.rect(0, A4[1] - 12 * mm, A4[0], 12 * mm, stroke=0, fill=1)
        canvas.setFont("R2AI-Bold", 7.8)
        canvas.setFillColor(colors.white)
        canvas.drawString(18 * mm, A4[1] - 7.7 * mm, "R2AI STAGE 3 · EXP143 TECHNICAL RELEASE")
        canvas.setStrokeColor(LINE)
        canvas.line(18 * mm, 14 * mm, A4[0] - 18 * mm, 14 * mm)
        canvas.setFont("R2AI", 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(18 * mm, 9.2 * mm, "stage3-exp143-20261103 · không phải hệ thống chẩn đoán")
        canvas.drawRightString(A4[0] - 18 * mm, 9.2 * mm, f"Trang {doc.page}")
        canvas.restoreState()


def cover(story, s, ui_image: Path | None):
    story.append(Spacer(1, 20 * mm))
    story.append(Paragraph("ROAD TO AI 2026 · STAGE 3", ParagraphStyle(
        "cover-kicker", fontName="R2AI-Bold", fontSize=10, leading=12, textColor=TEAL,
        alignment=TA_CENTER, spaceAfter=12)))
    story.append(Paragraph("THUYẾT MINH SẢN PHẨM", ParagraphStyle(
        "cover-title", fontName="R2AI-Bold", fontSize=28, leading=34, textColor=NAVY,
        alignment=TA_CENTER, spaceAfter=9)))
    story.append(Paragraph("Truy hồi tài liệu và bằng chứng y khoa đa ngôn ngữ", ParagraphStyle(
        "cover-sub", fontName="R2AI", fontSize=14, leading=18, textColor=MUTED,
        alignment=TA_CENTER, spaceAfter=14)))
    metrics = Table([
        ["0.3190", "0.3731", "0.2648"],
        ["Public Final", "Docs F2-macro", "Chunks F2-macro"],
    ], colWidths=[48 * mm] * 3, rowHeights=[14 * mm, 8 * mm])
    metrics.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BG), ("BOX", (0, 0), (-1, -1), .6, LINE),
        ("INNERGRID", (0, 0), (-1, -1), .4, LINE), ("FONT", (0, 0), (-1, 0), "R2AI-Bold", 17),
        ("FONT", (0, 1), (-1, 1), "R2AI", 7.5), ("TEXTCOLOR", (0, 0), (-1, 0), TEAL),
        ("TEXTCOLOR", (0, 1), (-1, 1), MUTED), ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(metrics)
    story.append(Spacer(1, 10 * mm))
    if ui_image and ui_image.exists():
        img = Image(str(ui_image), width=154 * mm, height=96.25 * mm)
        story.append(img)
        story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Release <b>stage3-exp143-20261103</b> · Điểm private chưa công bố", ParagraphStyle(
        "cover-meta", fontName="R2AI", fontSize=9, leading=12, textColor=MUTED,
        alignment=TA_CENTER)))
    story.append(Paragraph("&lt;PTIT/&gt; Phan Tuấn Quốc Anh", ParagraphStyle(
        "cover-team", fontName="R2AI-Bold", fontSize=10, leading=13, textColor=NAVY,
        alignment=TA_CENTER, spaceBefore=4)))
    story.append(PageBreak())


def parse_markdown(path: Path, s) -> list:
    lines = path.read_text(encoding="utf-8").splitlines()
    story: list = []
    paragraph: list[str] = []
    code: list[str] = []
    in_code = False

    def flush_paragraph():
        if paragraph:
            story.append(Paragraph(inline_markup(" ".join(paragraph)), s["body"]))
            paragraph.clear()

    def flush_code():
        if code:
            story.append(Paragraph("<br/>".join(escape(x) for x in code), s["code"]))
            code.clear()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_paragraph()
            if in_code:
                flush_code()
            in_code = not in_code
            continue
        if in_code:
            code.append(line)
            continue
        if not stripped:
            flush_paragraph()
            continue
        if stripped.startswith("# "):
            continue  # cover replaces the markdown H1
        if stripped.startswith("## "):
            flush_paragraph()
            story.append(Paragraph(inline_markup(stripped[3:]), s["h1"]))
            story.append(HRFlowable(width="100%", thickness=1, color=TEAL, spaceAfter=4))
            continue
        if stripped.startswith("### "):
            flush_paragraph()
            story.append(Paragraph(inline_markup(stripped[4:]), s["h2"]))
            continue
        if re.match(r"^\d+\.\d+\s", stripped):
            flush_paragraph()
            story.append(Paragraph(inline_markup(stripped), s["h3"]))
            continue
        if stripped.startswith(("- ", "* ")):
            flush_paragraph()
            story.append(Paragraph("• " + inline_markup(stripped[2:]), s["bullet"]))
            continue
        if re.match(r"^\d+\.\s", stripped):
            flush_paragraph()
            story.append(Paragraph(inline_markup(stripped), s["bullet"]))
            continue
        paragraph.append(stripped)
    flush_paragraph(); flush_code()
    return story


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "deliverables/stage3_20261104/technical_report.md")
    parser.add_argument("--output", type=Path, default=ROOT / "deliverables/stage3_20261104/technical_report.pdf")
    args = parser.parse_args()
    register_fonts(); s = styles()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    doc = NumberedDocTemplate(str(args.output), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
                              topMargin=19 * mm, bottomMargin=18 * mm,
                              title="R2AI Stage 3 — Exp143 Technical Report",
                              author="<PTIT/> Phan Tuấn Quốc Anh")
    story: list = []
    cover(story, s, ROOT / "deliverables/stage3_20261104/assets/ui_desktop.png")
    story.extend(parse_markdown(args.input, s))
    doc.build(story)
    print(args.output)


if __name__ == "__main__":
    main()
