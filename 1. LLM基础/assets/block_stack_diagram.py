#!/usr/bin/env python3
"""生成第11章 第九节"堆叠多个 Block"一节的结构图（原 Mermaid 图 6）：
Embedding --> Block × N --> 最终表示 --> 输出层，简单的四段水平链条。（高度压缩到 1/4）
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FONT_CANDIDATES = [
    "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC",
    "WenQuanYi Micro Hei", "Source Han Sans SC", "Arial Unicode MS",
]
available = {font.name for font in fm.fontManager.ttflist}
font = next((name for name in FONT_CANDIDATES if name in available), "DejaVu Sans")
plt.rcParams.update({"font.family": font, "axes.unicode_minus": False})


def draw_box(ax, cx, cy, w, h, text, facecolor="#EAF2FB", edgecolor="#1F5FA0", fontsize=9.5):
    box = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                          boxstyle="round,pad=0.015,rounding_size=0.05",
                          linewidth=1.4, edgecolor=edgecolor, facecolor=facecolor)
    ax.add_patch(box)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, weight="bold", color="#16324a")


def arrow(ax, start, end, color="#333333", lw=1.5):
    a = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=lw, color=color)
    ax.add_patch(a)


fig, ax = plt.subplots(figsize=(10.5, 0.62), dpi=170)

y = 0
boxes = [
    (0.8, 1.8, "Embedding", "#EAF2FB", "#1F5FA0"),
    (3.4, 2.2, "Block × N", "#FFF3D6", "#B98600"),
    (6.4, 2.0, "最终表示", "#F5EAF9", "#7A3FA0"),
    (9.1, 1.8, "输出层", "#E7F5EA", "#2E7D46"),
]
edges = []
for cx, w, text, fc, ec in boxes:
    draw_box(ax, cx, y, w, 0.5, text, facecolor=fc, edgecolor=ec)
    edges.append((cx - w / 2, cx + w / 2))

for (l0, r0), (l1, r1) in zip(edges[:-1], edges[1:]):
    arrow(ax, (r0 + 0.05, y), (l1 - 0.05, y))

ax.set_xlim(-0.5, 10.3)
ax.set_ylim(-0.32, 0.42)
ax.axis("off")
ax.set_title("多个 Transformer Block 堆叠后接输出层", fontsize=9.5, weight="bold", pad=2)

output = Path(__file__).resolve().parent / "block_stack_diagram.png"
fig.savefig(output, bbox_inches="tight")
print(f"saved to {output}")
