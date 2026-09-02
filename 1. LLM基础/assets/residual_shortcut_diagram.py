#!/usr/bin/env python3
"""生成第11章 第三节"为什么需要 Residual Connection"一节的结构图（原 Mermaid 图 1）：
x --> Attention --> + --> Output，同时 x 有一条虚线"直接跳过"支路直连到 +。
用 matplotlib 手绘方框与箭头，不依赖 Mermaid 渲染器。

修复记录：上一版把画布压得很扁，但虚线弧线的 rad 弯曲幅度没有跟着缩小，
导致弧顶超出可视区域被裁掉、中间出现断层。这一版改用"直角折线"画跳过支路
（从 x 上方走直线，转直角，再落到 + 上方），完全不依赖曲线弧度，
不会再出现因裁剪导致的断线问题；同时给 clip_on=False 做双重保险。
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
                          linewidth=1.4, edgecolor=edgecolor, facecolor=facecolor, clip_on=False)
    ax.add_patch(box)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, weight="bold",
            color="#16324a", clip_on=False)


def straight_arrow(ax, start, end, color="#333333", lw=1.5, ls="-"):
    a = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                         linewidth=lw, color=color, linestyle=ls, clip_on=False)
    ax.add_patch(a)


def elbow_bypass(ax, x_start, x_end, y_base, y_top, color="#B33A3A", lw=1.4):
    """画一条直角折线跳过支路：从 (x_start, y_base) 向上走到 y_top，
    水平走到 x_end 上方，再向下落回 (x_end, y_base)。不用曲线，避免裁剪断线。"""
    ax.plot([x_start, x_start], [y_base, y_top], color=color, lw=lw, ls="--", clip_on=False, zorder=1)
    ax.plot([x_start, x_end], [y_top, y_top], color=color, lw=lw, ls="--", clip_on=False, zorder=1)
    straight_arrow(ax, (x_end, y_top), (x_end, y_base + 0.05), color=color, lw=lw, ls="--")


fig, ax = plt.subplots(figsize=(10, 1.35), dpi=170)

y = 0
draw_box(ax, 0.6, y, 1.2, 0.55, "x")
draw_box(ax, 3.2, y, 2.4, 0.55, "Attention")
draw_box(ax, 6.2, y, 1.1, 0.55, "+", facecolor="#FFF3D6", edgecolor="#B98600")
draw_box(ax, 8.4, y, 1.8, 0.55, "Output", facecolor="#E7F5EA", edgecolor="#2E7D46")

straight_arrow(ax, (1.2, y), (2.0, y))
straight_arrow(ax, (4.4, y), (5.65, y))
straight_arrow(ax, (6.75, y), (7.5, y))

# 直角折线跳过支路：从 x 正上方走直线，绕过 Attention，落到 + 正上方
elbow_bypass(ax, 0.6, 6.2, y_base=0.28, y_top=0.62, color="#B33A3A", lw=1.4)
ax.text(3.4, 0.78, "直接跳过", fontsize=8, color="#B33A3A", ha="center", weight="bold", clip_on=False)

ax.set_xlim(-0.6, 9.6)
ax.set_ylim(-0.35, 1.0)
ax.axis("off")
ax.set_title("残差连接：实线走 Attention，虚线直接跳过", fontsize=9.5, weight="bold", pad=2)

output = Path(__file__).resolve().parent / "residual_shortcut_diagram.png"
fig.savefig(output, bbox_inches="tight")
print(f"saved to {output}")
