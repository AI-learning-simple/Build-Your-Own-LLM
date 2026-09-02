#!/usr/bin/env python3
"""生成第10章 Attention 完整流程图（单头 Scaled Dot-Product Attention）：
X 三路投影出 Q/K/V（备料）→ QK^T 点积 → ÷√d_k 缩放 →（可选 Causal Mask）→ Softmax → 权重 A（打分）
→ A·V 加权求和 → 输出（取内容）。
底部标注"备料 / 打分 / 取内容"三个阶段，与第三节的时间线叙述呼应。
横版扁平布局，全部使用直角折线连线（不依赖曲线弧度，避免画布压缩时断线）。
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


fig, ax = plt.subplots(figsize=(14.5, 3.7), dpi=170)

# ===== 布局 =====
y_Q, y_K, y_V = 0.95, 0.0, -0.95   # 三条投影支路的高度

# 输入框
draw_box(ax, 0.8, 0, 1.4, 0.85, "整句 Embedding\nX（N×d）")

# 三路投影框
draw_box(ax, 3.3, y_Q, 2.3, 0.6, "X @ W_Q → Q 表")
draw_box(ax, 3.3, y_K, 2.3, 0.6, "X @ W_K → K 表")
draw_box(ax, 3.3, y_V, 2.3, 0.6, "X @ W_V → V 表")

# 打分链（在 K 路所在的 y=0 主线上）
draw_box(ax, 5.8, 0, 1.6, 0.62, "QK^T\n点积打分")
draw_box(ax, 7.55, 0, 1.3, 0.62, "÷ √d_k\n缩放")
draw_box(ax, 9.2, 0, 1.5, 0.62, "Causal Mask\n（可选）", facecolor="#F4F4F4",
         edgecolor="#888888", dashed=True)
draw_box(ax, 10.8, 0, 1.2, 0.62, "Softmax")

# 取内容：A·V 与输出
draw_box(ax, 12.4, -0.5, 1.7, 0.72, "A · V\n加权求和",
         facecolor="#FFF3D6", edgecolor="#B98600")
draw_box(ax, 14.35, -0.5, 1.3, 0.72, "输出\n（新表示）",
         facecolor="#E7F5EA", edgecolor="#2E7D46")

# ===== 连线 =====
# X 右缘 → 竖直总线 → 三路投影
line(ax, [1.5, 1.9], [0, 0])
line(ax, [1.9, 1.9], [y_Q, y_V])
arrow(ax, (1.9, y_Q), (2.15, y_Q))
arrow(ax, (1.9, y_K), (2.15, y_K))
arrow(ax, (1.9, y_V), (2.15, y_V))

# Q 表 → 折线降到 QK^T 顶部
line(ax, [4.45, 5.8], [y_Q, y_Q])
arrow(ax, (5.8, y_Q), (5.8, 0.31))
# K 表 → 直进 QK^T 左缘
arrow(ax, (4.45, y_K), (5.0, y_K))

# 打分链内部箭头
arrow(ax, (6.6, 0), (6.9, 0))
arrow(ax, (8.2, 0), (8.45, 0))
arrow(ax, (9.95, 0), (10.2, 0))

# Softmax → 折线降到 A·V 顶部
line(ax, [11.4, 12.4], [0, 0])
arrow(ax, (12.4, 0), (12.4, -0.14))
# V 表 → 长水平线 → 升到 A·V 底部
line(ax, [4.45, 12.4], [y_V, y_V])
arrow(ax, (12.4, y_V), (12.4, -0.86))

# A·V → 输出
arrow(ax, (13.25, -0.5), (13.7, -0.5))

# ===== 底部三阶段标注 =====
def stage_bracket(ax, x0, x1, y, label, cx):
    line(ax, [x0, x1], [y, y], color="#888888", lw=1.0)
    line(ax, [x0, x0], [y - 0.07, y + 0.07], color="#888888", lw=1.0)
    line(ax, [x1, x1], [y - 0.07, y + 0.07], color="#888888", lw=1.0)
    ax.text(cx, y - 0.28, label, ha="center", va="top", fontsize=8.3,
            color="#555555", clip_on=False)

yb = -1.75
stage_bracket(ax, 0.1, 4.5, yb, "第 0 步：备料——三路投影，一次性完成", 2.3)
stage_bracket(ax, 5.0, 11.4, yb, "第 1 步：打分——只用 Q 表和 K 表", 8.2)
stage_bracket(ax, 11.9, 15.05, yb, "第 2 步：取内容——只用 V 表", 13.45)

ax.set_xlim(-0.3, 15.4)
ax.set_ylim(-2.35, 1.6)
ax.axis("off")
ax.set_title("Scaled Dot-Product Attention 完整流程：先备料、再打分、最后取内容",
             fontsize=12.5, weight="bold", pad=8)

output = Path(__file__).resolve().parent / "attention_full_pipeline_diagram.png"
fig.savefig(output, bbox_inches="tight")
print(f"saved to {output}")
