#!/usr/bin/env python3
"""将第 11 章 Markdown 导出为适合打印的 A4 PDF 与 300 DPI 页面 PNG。

针对 xhtml2pdf 中文排版的已知问题做了四项修复：
1. -pdf-word-wrap: CJK —— 启用 CJK 断行模式（关键修复）。默认模式下，没有空格的
   长中文串被视为一个不可断的"词"，会整体溢出行宽被裁剪：文字仍留在 PDF 文本流里
   （pypdf 能提取到），但视觉上被截断或与下一行内容相接，表现为"内容不完整、异常断行"。
2. 数字箭头（4 → 8）用不换行空格保护，避免在箭头前被拆行。
3. 特殊符号（勾、上标）替换为可检索写法，避免字体缺字变黑方块或空字符。
4. 生成后删除末尾空白页。
"""
from __future__ import annotations

import base64
import html
import io
import os
import re
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from markdown import markdown
from pypdf import PdfReader, PdfWriter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from xhtml2pdf import pisa


def choose_font() -> tuple[str, str]:
    candidates = [
        ("/System/Library/Fonts/STHeiti Medium.ttc", 0, "STHeitiMedium"),
        ("/System/Library/Fonts/STHeiti Light.ttc", 0, "STHeitiLight"),
        ("/Library/Fonts/Arial Unicode.ttf", 0, "ArialUnicode"),
    ]
    for path, index, name in candidates:
        if Path(path).exists():
            pdfmetrics.registerFont(TTFont(name, path, subfontIndex=index))
            return path, name
    raise FileNotFoundError("未找到可嵌入的 CJK 字体。")


def mermaid_png(code: str) -> bytes:
    with tempfile.NamedTemporaryFile("w", suffix=".mmd", encoding="utf-8", delete=False) as f:
        f.write(code)
        source = f.name
    target = source[:-4] + ".png"
    try:
        subprocess.run(["mmdc", "-i", source, "-o", target, "-w", "1400", "-b", "white"],
                       check=True, capture_output=True, timeout=60)
        return Path(target).read_bytes()
    finally:
        for path in (source, target):
            if os.path.exists(path):
                os.unlink(path)


def math_image(source: str) -> str:
    source = source.strip().replace(r"\lVert", r"\|").replace(r"\rVert", r"\|")
    source = source.replace(r"\text{Attention Score}", r"\mathrm{Attention\ Score}")
    fig = plt.figure(figsize=(2.1, 0.24), dpi=220)
    fig.text(0.02, 0.5, "$" + source + "$", fontsize=7.2, va="center", color="#111")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    return '<img class="formula" src="data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode() + '"/>'


SUPERSCRIPT = str.maketrans({
    "⁰": "^0", "¹": "^1", "²": "^2", "³": "^3", "⁴": "^4",
    "⁵": "^5", "⁶": "^6", "⁷": "^7", "⁸": "^8", "⁹": "^9", "⁻": "^-",
})
SUBSCRIPT = str.maketrans({
    "₀": "_0", "₁": "_1", "₂": "_2", "₃": "_3", "₄": "_4",
    "₅": "_5", "₆": "_6", "₇": "_7", "₈": "_8", "₉": "_9", "ᵢ": "_i",
})


def render_markdown(source_path: Path) -> str:
    source = source_path.read_text(encoding="utf-8")
    source = source.replace("✅", "[OK]").replace("✓", "允许")
    source = source.replace("ᵀ", "^T").replace("√", "sqrt(")
    source = source.translate(SUPERSCRIPT).translate(SUBSCRIPT)
    source = re.sub(r"(?<=\d) → (?=\d)", "\u00a0→\u00a0", source)

    source = re.sub(r"```math\n(.*?)```", lambda m: math_image(m.group(1)), source, flags=re.S)

    def mermaid(m):
        try:
            encoded = base64.b64encode(mermaid_png(m.group(1).strip())).decode()
            return f'<img class="fig mermaid-fig" src="data:image/png;base64,{encoded}"/>'
        except Exception as exc:
            print("[WARN] Mermaid 渲染失败:", exc)
            return "<pre>" + html.escape(m.group(1)) + "</pre>"

    source = re.sub(r"```mermaid\n(.*?)```", mermaid, source, flags=re.S)

    def image(m):
        path = m.group(2)
        if path.startswith("http"):
            return m.group(0)
        file_path = Path(path) if Path(path).is_absolute() else source_path.parent / path
        if not file_path.exists():
            return m.group(0)
        suffix = file_path.suffix.lower().lstrip(".") or "png"
        encoded = base64.b64encode(file_path.read_bytes()).decode()
        return f'<img class="fig" src="data:image/{suffix};base64,{encoded}"/>'

    source = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", image, source)
    body = markdown(source, extensions=["tables", "fenced_code", "toc"])

    def code_lines(m):
        content = m.group(2).replace("\n", "<br/>\n").replace("  ", "&nbsp; ")
        return f'<pre><code{m.group(1)}>{content}</code></pre>'

    body = re.sub(r"<pre><code([^>]*)>(.*?)</code></pre>", code_lines, body, flags=re.S)
    font_path, font_name = choose_font()
    css = f"""
    @font-face {{ font-family: {font_name}; src: url("{font_path}"); }}
    @page {{ size: A4; margin: 1cm; }}
    body {{ font-family: {font_name}; font-size: 8.3pt; line-height: 1.22; color:#111;
           -pdf-word-wrap: CJK; }}
    p, li, td, th, blockquote, h1, h2, h3, pre {{ -pdf-word-wrap: CJK; }}
    h1 {{ font-size: 15pt; margin: 4pt 0 3pt; -pdf-keep-with-next: true; }}
    h2 {{ font-size: 11.5pt; margin: 7pt 0 2pt; -pdf-keep-with-next: true; }}
    h3 {{ font-size: 9.8pt; margin: 5pt 0 2pt; -pdf-keep-with-next: true; }}
    p {{ margin: 2pt 0; }}
    code, pre {{ font-family: {font_name}; font-size: 7.3pt; white-space: pre-wrap; }}
    pre {{ padding: 3px 4px; border: 0.5pt solid #ddd; background:#f4f4f4; }}
    .fig {{ max-width: 82%; height: auto; display:block; margin: 2pt auto; }}
    .mermaid-fig {{ width:auto; height:4.5cm; max-width:70%; max-height:4.5cm; object-fit:contain; }}
    .formula {{ width:4cm; max-width:4cm; height:auto; display:block; margin:1pt auto; }}
    table {{ border-collapse:collapse; font-size:7.8pt; width:100%; }}
    td, th {{ border:0.5pt solid #888; padding:1px 3px; }}
    blockquote {{ border-left:2pt solid #bbb; margin:3pt 0; padding-left:7pt; color:#444; }}
    ul, ol {{ margin:3pt 0; padding-left:16pt; }}
    """
    return f"<html><head><style>{css}</style></head><body>{body}</body></html>"


def remove_trailing_blank_pages(pdf_path: Path) -> None:
    reader = PdfReader(str(pdf_path))
    if len(reader.pages) <= 1:
        return
    keep = []
    for page in reader.pages:
        text = (page.extract_text() or "").strip()
        resources = page.get("/Resources")
        xobjects = resources.get("/XObject") if resources else None
        if text or xobjects or not keep:
            keep.append(page)
    if len(keep) == len(reader.pages):
        return
    writer = PdfWriter()
    for page in keep:
        writer.add_page(page)
    with pdf_path.open("wb") as output:
        writer.write(output)


def main() -> None:
    source = Path(__file__).resolve().parent / "11. Transformer Block.md"
    pdf_path = source.with_name("11. Transformer Block A4.pdf")
    png_dir = source.with_name("11. Transformer Block A4 PNG")
    document = render_markdown(source)
    with pdf_path.open("wb") as output:
        result = pisa.CreatePDF(document, dest=output)
    if result.err:
        raise RuntimeError("PDF 生成失败: " + str(result.log))
    remove_trailing_blank_pages(pdf_path)
    print("PDF:", pdf_path)
    try:
        import fitz
        pdf = fitz.open(pdf_path)
        png_dir.mkdir(parents=True, exist_ok=True)
        for old in png_dir.glob("*.png"):
            old.unlink()
        matrix = fitz.Matrix(300 / 72, 300 / 72)
        for index, page in enumerate(pdf, 1):
            page.get_pixmap(matrix=matrix, alpha=False).save(png_dir / f"page-{index:02d}.png")
        print("PNG:", png_dir, "pages:", len(pdf), "dpi: 300")
    except ImportError:
        print("[WARN] 未安装 PyMuPDF，跳过 PNG 输出。")


if __name__ == "__main__":
    main()
