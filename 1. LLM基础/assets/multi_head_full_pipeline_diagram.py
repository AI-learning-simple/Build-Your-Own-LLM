#!/usr/bin/env python3
"""生成第10章 Multi-Head Attention 完整流程图：
输入 X 复制到每个头 → 每个头用自己独立的 W_Q/W_K/W_V 投影并算一遍单头
Scaled Dot-Product Attention → H 份输出沿特征维 Concat 拼接 → 过输出投影 W_O → 最终输出。
横版扁平布局，总线式连线（竖直总线 + 水平支线），不依赖曲线弧度。
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


def draw_box(ax, cx, cy, w, h, text, facecolor="#EAF2FB", edgecolor="#1F5FA0",
             fontsize=8.5, dashed=False):
    box = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                          boxstyle="round,pad=0.02,rounding_size=0.05",
                          linewidth=1.5, edgecolor=edgecolor, facecolor=facecolor,
                          clip_on=False, linestyle="--" if dashed else "-")
    ax.add_patch(box)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, weight="bold",
            color="#16324a", clip_on=False)


def arrow(ax, start, end, color="#333333", lw=1.4, ls="-"):
    a = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11,
                         linewidth=lw, color=color, linestyle=ls, clip_on=False)
    ax.add_patch(a)


def line(ax, xs, ys, color="#333333", lw=1.4):
    ax.plot(xs, ys, color=color, lw=lw, clip_on=False, zorder=1)


fig, ax = plt.subplots(figsize=(13.5, 3.6), dpi=170)

y_H1, y_H2, y_Hh = 0.95, 0.0, -0.95   # 三个头的高度

# 输入框
draw_box(ax, 0.9, 0, 1.6, 0.85, "输入 X\n(B, T, d_model)")

# 三个头（头 h 用虚线表示"第 h 个，共 H 个"）
draw_box(ax, 4.3, y_H1, 4.0, 0.72, "头 1\nQ1K1^T/√d_k → Softmax → 加权 V1")
draw_box(ax, 4.3, y_H2, 4.0, 0.72, "头 2\nQ2K2^T/√d_k → Softmax → 加权 V2")
draw_box(ax, 4.3, y_Hh, 4.0, 0.72, "头 h（共 H 个头并行）\nQhKh^T/√d_k → Softmax → 加权 Vh",
         dashed=True, facecolor="#F6FAFD")
# 省略号：头 2 与头 h 之间
ax.text(4.3, -0.48, "……", ha="center", va="center", fontsize=10, color="#888888",
        clip_on=False)

# Concat / W_O / 输出
draw_box(ax, 8.4, 0, 1.8, 0.78, "Concat 拼接\n(B, T, H·d_head)",
         facecolor="#FFF3D6", edgecolor="#B98600")
draw_box(ax, 10.5, 0, 1.6, 0.78, "× W_O\n输出投影")
draw_box(ax, 12.5, 0, 1.7, 0.78, "输出\n(B, T, d_model)",
         facecolor="#E7F5EA", edgecolor="#2E7D46")

# ===== 连线 =====
# X 右缘 → 竖直总线 → 三个头
line(ax, [1.7, 2.1], [0, 0])
line(ax, [2.1, 2.1], [y_H1, y_Hh])
arrow(ax, (2.1, y_H1), (2.3, y_H1))
arrow(ax, (2.1, y_H2), (2.3, y_H2))
arrow(ax, (2.1, y_Hh), (2.3, y_Hh))

# 三个头 → 右侧竖直总线 → Concat
line(ax, [6.3, 7.0], [y_H1, y_H1])
line(ax, [6.3, 7.0], [y_H2, y_H2])
line(ax, [6.3, 7.0], [y_Hh, y_Hh])
line(ax, [7.0, 7.0], [y_H1, y_Hh])
arrow(ax, (7.0, 0), (7.5, 0))

# Concat → W_O → 输出
arrow(ax, (9.3, 0), (9.7, 0))
arrow(ax, (11.3, 0), (11.65, 0))

# 底部注释
ax.text(6.3, -1.75,
        "每个头拥有自己独立的一套 W_Q / W_K / W_V 投影矩阵（图中省略上标）；\n"
        "每个头算出各自的 C，拼接只是把它们接回 d_model 宽度，W_O 负责融合成模块最终输出。",
        ha="center", va="top", fontsize=8.3, color="#555555", clip_on=False)

ax.set_xlim(-0.3, 13.6)
ax.set_ylim(-2.5, 1.6)
ax.axis("off")
ax.set_title("Multi-Head Attention 完整流程：H 个头并行计算，拼接后过输出投影 W_O",
             fontsize=12.5, weight="bold", pad=8)

output = Path(__file__).resolve().parent / "multi_head_full_pipeline_diagram.png"
fig.savefig(output, bbox_inches="tight")
print(f"saved to {output}")
