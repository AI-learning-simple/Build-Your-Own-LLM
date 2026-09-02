#!/usr/bin/env python3
"""生成第11章 第四节 Pre-LN 结构图（原 Mermaid 图 3）：
x --> LayerNorm --> Attention --> Residual Add，同时 x 有一条虚线"恒等支路"
绕过 LayerNorm 和 Attention 两个子层，直接连到 Residual Add。

修复记录：改用直角折线画恒等支路（不用曲线弧度），避免因画布压扁后弧顶被裁剪导致断线。
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


def draw_box(ax, cx, cy, w, h, text, facecolor="#EAF2FB", edgecolor="#1F5FA0", fontsize=9):
    box = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                          boxstyle="round,pad=0.015,rounding_size=0.05",
                          linewidth=1.4, edgecolor=edgecolor, facecolor=facecolor, clip_on=False)
    ax.add_patch(box)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, weight="bold",
            color="#16324a", clip_on=False)


def straight_arrow(ax, start, end, color="#333333", lw=1.5, ls="-"):
    a = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                         linewidth=lw, color=color, linestyle=ls, clip_on=False)
    ax.add_patch(a)


def elbow_bypass(ax, x_start, x_end, y_base, y_top, color="#B33A3A", lw=1.4):
    ax.plot([x_start, x_start], [y_base, y_top], color=color, lw=lw, ls="--", clip_on=False, zorder=1)
    ax.plot([x_start, x_end], [y_top, y_top], color=color, lw=lw, ls="--", clip_on=False, zorder=1)
    straight_arrow(ax, (x_end, y_top), (x_end, y_base + 0.05), color=color, lw=lw, ls="--")


fig, ax = plt.subplots(figsize=(11.5, 1.4), dpi=170)

y = 0
draw_box(ax, 0.6, y, 1.2, 0.55, "x")
draw_box(ax, 3.0, y, 2.0, 0.55, "LayerNorm")
draw_box(ax, 5.6, y, 2.2, 0.55, "Attention")
draw_box(ax, 8.6, y, 2.6, 0.55, "Residual Add", facecolor="#FFF3D6", edgecolor="#B98600")

straight_arrow(ax, (1.2, y), (2.0, y))
straight_arrow(ax, (4.0, y), (4.5, y))
straight_arrow(ax, (6.7, y), (7.3, y))

# 直角折线恒等支路：从 x 正上方走直线，绕过 LayerNorm 和 Attention，落到 Residual Add 正上方
elbow_bypass(ax, 0.6, 8.6, y_base=0.28, y_top=0.62, color="#B33A3A", lw=1.4)
ax.text(4.6, 0.78, "恒等支路", fontsize=8, color="#B33A3A", ha="center", weight="bold", clip_on=False)

ax.set_xlim(-0.6, 11.2)
ax.set_ylim(-0.35, 1.0)
ax.axis("off")
ax.set_title("Pre-LN 结构：先归一化，再计算；虚线恒等支路绕过整个子层", fontsize=9.5, weight="bold", pad=2)

output = Path(__file__).resolve().parent / "preln_residual_diagram.png"
fig.savefig(output, bbox_inches="tight")
print(f"saved to {output}")
