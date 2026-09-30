"""Build the paginated results PDF from reviewable Markdown and evidence plots."""

import re
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

import charts
import measured_charts

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUTPUT = ROOT / "output/pdf/atfm-gpu-results-2026-09-30.pdf"
WIDTH = 499
INK = colors.HexColor("#23364d")


def styles():
    fonts = Path("/usr/share/fonts/truetype/dejavu")
    for name, file in [("Body", "DejaVuSans.ttf"), ("Bold", "DejaVuSans-Bold.ttf"), ("Code", "DejaVuSansMono.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(fonts / file)))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Bold", italic="Body", boldItalic="Bold")
    base = dict(fontName="Body", fontSize=9.3, leading=14.1, textColor=INK, spaceAfter=9)
    result = {"body": ParagraphStyle("body", **base)}
    for name, size, leading in [("h1", 23, 28), ("h2", 14, 19), ("h3", 11.5, 16)]:
        result[name] = ParagraphStyle(name, fontName="Bold", fontSize=size, leading=leading,
                                      textColor=INK, spaceAfter=12, spaceBefore=6)
    result["cell"] = ParagraphStyle("cell", parent=result["body"], fontSize=8, leading=11.2, spaceAfter=0)
    return result


def inline(text):
    text = escape(text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<link href="\2" color="#3066a5">\1</link>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    return re.sub(r"`([^`]+)`", r'<font name="Code" size="8">\1</font>', text)


def table(block, style):
    rows = [line.strip("|").split("|") for line in block.splitlines() if not re.match(r"\|[\s:|\-]+$", line)]
    cells = [[Paragraph(inline(cell.strip()), style) for cell in row] for row in rows]
    count = len(cells[0])
    ratios = {3: [.28, .29, .43], 4: [.24, .25, .25, .26], 5: [.18, .12, .19, .2, .31]}
    widths = [WIDTH * value for value in ratios.get(count, [1 / count] * count)]
    result = Table(cells, colWidths=widths, hAlign="LEFT", repeatRows=1)
    result.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe9f3")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f4f7fa"), colors.white]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7), ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    return [result, Spacer(1, 12)]


def figure(block):
    name = re.search(r"\]\(([^)]+)\)", block).group(1)
    path = HERE / name
    with PILImage.open(path) as image:
        width, height = image.size
    return Image(str(path), width=WIDTH, height=WIDTH * height / width)


def blocks(page, style):
    result = []
    for block in page.strip().split("\n\n"):
        if block.startswith("|"):
            result.extend(table(block, style["cell"]))
        elif block.startswith("!["):
            result.extend([figure(block), Spacer(1, 8)])
        elif block.startswith("#"):
            heading, text = block.split(" ", 1)
            result.append(Paragraph(inline(text), style[f"h{len(heading)}"]))
        else:
            result.append(Paragraph(inline(block.replace("\n", " ")), style["body"]))
    return result


def footer(canvas, doc):
    canvas.setStrokeColor(colors.HexColor("#cbd5df"))
    canvas.line(48, 806, 547, 806)
    canvas.setFont("Body", 7.5)
    canvas.setFillColor(INK)
    canvas.drawString(48, 815, "ATFM  /  GPU EVIDENCE & CACHING TRADEOFFS")
    canvas.drawRightString(547, 815, "30 SEPTEMBER 2026")
    canvas.drawString(48, 28, "Measured results, derived bounds, and illustrative scenarios are labeled separately.")
    canvas.drawRightString(547, 28, str(doc.page))


def build():
    charts.build()
    measured_charts.build()
    style, story = styles(), []
    pages = (HERE / "report.md").read_text().split("---PAGE---")
    for i, page in enumerate(pages):
        if i:
            story.append(PageBreak())
        story.extend(blocks(page, style))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=(595, 842), leftMargin=48, rightMargin=48,
                            topMargin=54, bottomMargin=48, title="ATFM GPU Results and Caching Tradeoffs",
                            author="ATFM project", pageCompression=1, invariant=1)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    (HERE / OUTPUT.name).write_bytes(OUTPUT.read_bytes())
    print(OUTPUT)


if __name__ == "__main__":
    build()
