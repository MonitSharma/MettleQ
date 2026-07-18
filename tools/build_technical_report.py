#!/usr/bin/env python3
"""Build the MettleQ paper-style PDF handout from frozen evidence."""

from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, KeepTogether, PageTemplate, Paragraph,
    Spacer, Table, TableStyle, PageBreak,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper" / "MettleQ_technical_report.md"
OUTPUT = ROOT / "output" / "pdf" / "MettleQ_technical_report.pdf"
CHART = ROOT / "assets" / "benchmarks-frozen" / "fork-m3pro-20260718-sdk-crossover-radix16-idle" / "sdk_cpu_gpu_crossover.png"
LIMIT_CHART = ROOT / "assets" / "benchmarks-frozen" / "fork-m3pro-20260718-safe-dense-limits-radix16-idle" / "safe_dense_limit.png"


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D9DEE8"))
    canvas.line(18 * mm, 14 * mm, 192 * mm, 14 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#5D6675"))
    canvas.drawString(18 * mm, 9 * mm, "MettleQ technical report - version 0.2")
    canvas.drawRightString(192 * mm, 9 * mm, f"Page {doc.page}")
    canvas.restoreState()


def _inline(text):
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r"<font name='Courier'>\1</font>", text)
    return text


def _table(rows, widths):
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#18243A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F6FA")]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C7CEDA")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def build():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="PaperTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=23, leading=28, alignment=TA_CENTER,
        textColor=colors.HexColor("#16233B"), spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="Subtitle", parent=styles["Normal"], fontSize=10.5, leading=15,
        alignment=TA_CENTER, textColor=colors.HexColor("#596477"), spaceAfter=18,
    ))
    styles.add(ParagraphStyle(
        name="Section", parent=styles["Heading1"], fontName="Helvetica-Bold",
        fontSize=15, leading=19, textColor=colors.HexColor("#234A91"),
        spaceBefore=13, spaceAfter=7, keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        name="BodyPaper", parent=styles["BodyText"], fontSize=9.2, leading=13.2,
        textColor=colors.HexColor("#202938"), spaceAfter=7,
    ))
    styles.add(ParagraphStyle(
        name="BulletPaper", parent=styles["BodyText"], fontSize=9, leading=12.7,
        leftIndent=12, firstLineIndent=-7, bulletIndent=4, spaceAfter=3,
    ))
    styles.add(ParagraphStyle(
        name="Abstract", parent=styles["BodyText"], fontSize=9.3, leading=13.5,
        leftIndent=8, rightIndent=8, borderColor=colors.HexColor("#D4DBE7"),
        borderWidth=0.7, borderPadding=9, backColor=colors.HexColor("#F6F8FC"),
        spaceAfter=12,
    ))

    doc = BaseDocTemplate(
        str(OUTPUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=17 * mm, bottomMargin=19 * mm,
        title="MettleQ: A Trustworthy Native Apple-Silicon Backend for Quantum Simulation",
        author="MettleQ project",
        subject="Architecture, validation, and Apple Silicon benchmark report",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="paper")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=_footer)])

    text = SOURCE.read_text()
    lines = text.splitlines()
    story = []
    paragraph = []
    abstract_mode = False
    table_rows = []

    def flush_paragraph():
        nonlocal paragraph
        if paragraph:
            value = _inline(" ".join(item.strip() for item in paragraph))
            style = styles["Abstract"] if abstract_mode else styles["BodyPaper"]
            story.append(Paragraph(value, style))
            paragraph = []

    def flush_table():
        nonlocal table_rows
        if table_rows:
            clean = [[cell.strip() for cell in row] for row in table_rows]
            columns = len(clean[0])
            widths = [doc.width / columns] * columns
            story.append(Spacer(1, 3))
            story.append(_table(clean, widths))
            story.append(Spacer(1, 8))
            table_rows = []

    for line in lines:
        if line.startswith("# "):
            flush_paragraph(); flush_table()
            story.append(Spacer(1, 18 * mm))
            story.append(Paragraph(_inline(line[2:]), styles["PaperTitle"]))
        elif line.startswith("**Technical report"):
            flush_paragraph()
            story.append(Paragraph(_inline(line), styles["Subtitle"]))
        elif line.startswith("## "):
            flush_paragraph(); flush_table()
            abstract_mode = line[3:].strip() == "Abstract"
            story.append(Paragraph(_inline(line[3:]), styles["Section"]))
        elif line.startswith("| "):
            flush_paragraph()
            cells = line.strip().strip("|").split("|")
            if all(set(cell.strip()) <= {"-", ":"} for cell in cells):
                continue
            table_rows.append(cells)
        elif line.startswith("- ") or re.match(r"\d+\. ", line):
            flush_paragraph(); flush_table()
            content = re.sub(r"^(?:- |\d+\. )", "", line)
            story.append(Paragraph("- " + _inline(content), styles["BulletPaper"]))
        elif not line.strip():
            flush_paragraph(); flush_table()
        else:
            paragraph.append(line)
    flush_paragraph(); flush_table()

    # Insert sharp frozen charts before the limitations section.
    insert_at = next(
        (i for i, item in enumerate(story)
         if isinstance(item, Paragraph) and item.getPlainText().startswith("8. Capacity")),
        len(story),
    )
    charts = [
        PageBreak(),
        Paragraph("Measured crossover and guarded capacity", styles["Section"]),
        Image(str(CHART), width=164 * mm, height=125 * mm),
        Spacer(1, 7),
        Paragraph(
            "Figure 1. Complete full-state CPU/GPU crossover. Ratios above one favor MettleQ; annotations include maximum phase-aligned state error.",
            styles["BodyPaper"],
        ),
        Image(str(LIMIT_CHART), width=150 * mm, height=82 * mm),
        Paragraph(
            "Figure 2. Monitored scalar-output capacity probe. The 30-qubit marks are pre-launch safety refusals.",
            styles["BodyPaper"],
        ),
        PageBreak(),
    ]
    story[insert_at:insert_at] = charts
    doc.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    build()
