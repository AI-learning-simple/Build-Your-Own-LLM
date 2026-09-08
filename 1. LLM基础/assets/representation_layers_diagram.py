#!/usr/bin/env python3
"""生成第11章 第十节"表示学习"一节的链条图（原 Mermaid 图）：
Token → Embedding v0 → Layer 1 v1 → Layer 2 v2 → ... vn → Next Token Prediction。
展示 Token 的表示随层数逐层演变。
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


fig, ax = plt.subplots(figsize=(12.5, 0.7), dpi=170)

y = 0
# 从"输入"到"预测"，颜色由冷到暖再回到绿，体现表示逐层深化
boxes = [
    (1.05, 1.5, "Token", "#EAF2FB", "#1F5FA0"),
    (3.35, 2.3, "Embedding v0", "#EAF2FB", "#1F5FA0"),
    (5.90, 2.0, "Layer 1 v1", "#FFF3D6", "#B98600"),
    (8.30, 2.0, "Layer 2 v2", "#FDEBD0", "#C06900"),
    (10.40, 1.4, "... vn", "#F5EAF9", "#7A3FA0"),
    (13.10, 3.2, "Next Token Prediction", "#E7F5EA", "#2E7D46"),
]
edges = []
for cx, w, text, fc, ec in boxes:
    draw_box(ax, cx, y, w, 0.52, text, facecolor=fc, edgecolor=ec)
    edges.append((cx - w / 2, cx + w / 2))

for (l0, r0), (l1, r1) in zip(edges[:-1], edges[1:]):
    arrow(ax, (r0 + 0.05, y), (l1 - 0.05, y))

ax.set_xlim(-0.2, 15.2)
ax.set_ylim(-0.36, 0.46)
ax.axis("off")
ax.set_title("Token 的表示随层数逐层演变", fontsize=10, weight="bold", pad=2)

output = Path(__file__).resolve().parent / "representation_layers_diagram.png"
fig.savefig(output, bbox_inches="tight")
print(f"saved to {output}")
