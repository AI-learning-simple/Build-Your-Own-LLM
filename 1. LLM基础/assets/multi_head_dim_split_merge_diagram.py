#!/usr/bin/env python3
"""生成第 10 章 Multi-Head 维度拆分与合并示意图（d_model=4、H=2 玩具例子）。
单 Token 视角：4 维向量沿特征维切成两半 → 头 1 / 头 2 各自独立算 Attention →
沿特征维横向拼回 4 维 → 过输出投影 W_O 融合。横版扁平布局，直角折线连线。
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

FONT_CANDIDATES = [
    "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC",
    "WenQuanYi Micro Hei", "Source Han Sans SC", "Arial Unicode MS",
]
available = {font.name for font in fm.fontManager.ttflist}
font = next((name for name in FONT_CANDIDATES if name in available), "DejaVu Sans")
plt.rcParams.update({"font.family": font, "axes.unicode_minus": False})

BLUE_F, BLUE_E = "#DCEAF8", "#1F5FA0"
ORANGE_F, ORANGE_E = "#FDEBD8", "#C56A0F"
GREEN_F, GREEN_E = "#E2F3E6", "#2E7D46"
INK = "#16324a"


def cell(ax, x, y, w, h, text, fc, ec, fs=9):
    """以 (x, y) 为左下角画一格并居中写数字。"""
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=1.4, clip_on=False))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=INK, clip_on=False, weight="bold")


def box(ax, cx, cy, w, h, text, fc="#F6FAFD", ec="#1F5FA0", fs=8.5):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.05", lw=1.4,
                                edgecolor=ec, facecolor=fc, clip_on=False))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=INK,
            weight="bold", clip_on=False)


def elbow(ax, pts, color="#333333", lw=1.3, arrow_end=False):
    """直角折线，可选取末端箭头。"""
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        ax.plot([x1, x2], [y1, y2], color=color, lw=lw, clip_on=False,
                solid_capstyle="round", zorder=1)
    if arrow_end:
        ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>",
                                     mutation_scale=10, lw=lw, color=color, clip_on=False))


fig, ax = plt.subplots(figsize=(15, 3.9), dpi=170)

CW, CH = 1.15, 1.0  # 格子尺寸
Y1, Y2 = 2.05, -2.05  # 头 1 / 头 2 路线高度

# ===== 左：cat 的 4 维向量（前 2 维蓝 = 头 1 份额，后 2 维橙 = 头 2 份额）=====
x0 = 0.3
for i, v in enumerate(["0.9", "0.6", "-0.3", "0.8"]):
    fc, ec = (BLUE_F, BLUE_E) if i < 2 else (ORANGE_F, ORANGE_E)
    cell(ax, x0 + i * CW, -CH / 2, CW, CH, v, fc, ec)
ax.text(x0 + 2 * CW + 0.3, 0.95, "cat 的向量（d_model = 4）", ha="center", va="bottom",
        fontsize=9, weight="bold", color=INK, clip_on=False)
# 切分虚线
ax.plot([x0 + 2 * CW] * 2, [-CH / 2 - 0.12, CH / 2 + 0.12], color="#C0392B",
        lw=1.6, ls="--", clip_on=False)
ax.text(x0 + 2 * CW, -0.95, "沿特征维切开\n（不是按 Token 分）", ha="center", va="top",
        fontsize=7.5, color="#C0392B", clip_on=False)

# ===== 两条支路：头 1（上，蓝）/ 头 2（下，橙）=====
for y, idx, head, nums, out, fc, ec in [
    (Y1, "1", "头 1：每个 Token 的前 2 维", ["0.9", "0.6"], ["c1", "c2"], BLUE_F, BLUE_E),
    (Y2, "2", "头 2：每个 Token 的后 2 维", ["-0.3", "0.8"], ["c3", "c4"], ORANGE_F, ORANGE_E),
]:
    xin = 5.0
    for i, v in enumerate(nums):
        cell(ax, xin + i * CW, y - CH / 2, CW, CH, v, fc, ec)
    ax.text(xin + len(nums) * CW / 2, y + CH / 2 + 0.12, head, ha="center", va="bottom",
            fontsize=7.5, color=ec, weight="bold", clip_on=False)
    box(ax, 10.2, y, 4.3, 0.95,
        f"Scaled Dot-Product Attention\nQ{idx}K{idx}^T/√d_head → Softmax → ×V{idx}",
        fs=7.5)
    xout = 12.9
    for i, v in enumerate(out):
        cell(ax, xout + i * CW, y - CH / 2, CW, CH, v, fc, ec)
    # 支路内部箭头
    arrow_kwargs = dict(color="#333333", lw=1.2)
    ax.add_patch(FancyArrowPatch((xin + len(nums) * CW + 0.1, y), (8.05, y),
                                 arrowstyle="-|>", mutation_scale=10, **arrow_kwargs))
    ax.add_patch(FancyArrowPatch((12.45, y), (xout - 0.1, y),
                                 arrowstyle="-|>", mutation_scale=10, **arrow_kwargs))

# ===== 左条 → 两条支路的直角折线 =====
# 头 1（第 1 格中心）直线向上；头 2 绕开条下方的红字标签再下来
elbow(ax, [(x0 + CW / 2, CH / 2), (x0 + CW / 2, Y1), (xin - 0.15, Y1)], arrow_end=True)
elbow(ax, [(x0 + 3 * CW / 2, -CH / 2), (x0 + 3 * CW / 2, -0.78), (4.3, -0.78), (4.3, Y2),
           (xin - 0.15, Y2)], arrow_end=True)

# ===== 右：Concat 拼回 4 维 → W_O → 最终输出 =====
xc = 16.0
for i, (v, (fc, ec)) in enumerate(zip(
        ["c1", "c2", "c3", "c4"],
        [(BLUE_F, BLUE_E)] * 2 + [(ORANGE_F, ORANGE_E)] * 2)):
    cell(ax, xc + i * CW, -CH / 2, CW, CH, v, fc, ec)
ax.text(xc + 2 * CW, 0.95, "Concat：沿特征维横着拼（Token 数不变）\n(B,T,2) 与 (B,T,2) → (B,T,4)",
        ha="center", va="bottom", fontsize=7.5, weight="bold", color="#B98600", clip_on=False)

box(ax, 22.4, 0, 1.7, 1.05, "× W_O\n（融合两头）", fc="#FFF3D6", ec="#B98600", fs=8)

xf = 24.2
for i, v in enumerate(["out1", "out2", "out3", "out4"]):
    cell(ax, xf + i * CW, -CH / 2, CW, CH, v, GREEN_F, GREEN_E)
ax.text(xf + 2 * CW, 0.95, "最终输出\n(B, T, 4) = (B, T, d_model)", ha="center", va="bottom",
        fontsize=7.5, weight="bold", color=GREEN_E, clip_on=False)

# 支路输出 → Concat（头 1 接左半、头 2 接右半）
elbow(ax, [(12.9 + 2 * CW + 0.1, Y1), (15.5, Y1), (15.5, 0.18), (xc - 0.15, 0.18)], arrow_end=True)
elbow(ax, [(12.9 + 2 * CW + 0.1, Y2), (15.5, Y2), (15.5, -0.18), (xc - 0.15, -0.18)], arrow_end=True)
# Concat → W_O → 输出
ax.add_patch(FancyArrowPatch((xc + 4 * CW + 0.1, 0), (21.5, 0),
                             arrowstyle="-|>", mutation_scale=10, lw=1.2, color="#333333"))
ax.add_patch(FancyArrowPatch((23.3, 0), (xf - 0.15, 0),
                             arrowstyle="-|>", mutation_scale=10, lw=1.2, color="#333333"))

# ===== 底注 =====
ax.text(13.5, -3.15,
        "两个头看到的 Token 完全相同，被切开的是每个 Token 向量的内部；拼接时每个 Token 从 d_head=2 拼回 d_model=4，"
        "再由 W_O 把两个角度融合成一份新表示。",
        ha="center", va="top", fontsize=7.8, color="#555555", clip_on=False)

ax.set_xlim(-0.2, 29.2)
ax.set_ylim(-3.7, 3.3)  # 必须覆盖头 1 标签顶(~2.9)，否则总标题会与超界内容重叠
ax.axis("off")
ax.set_title("Multi-Head 的维度拆分与合并（玩具例子：d_model=4、H=2、d_head=2）",
             fontsize=11.5, weight="bold", pad=6)

output = Path(__file__).resolve().parent / "multi_head_dim_split_merge_diagram.png"
fig.savefig(output, bbox_inches="tight")
print(f"saved to {output}")
