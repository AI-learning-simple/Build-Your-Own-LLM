#!/usr/bin/env python3
"""生成第11章 第五节"训练时故意捣乱"一节的结构图（原 Mermaid 图 4）：
Input 扇出到 4 个神经元（2 个关闭），再扇入汇总到 Output。

修复记录：上一版把 4 个神经元排成 2x2 网格后，"输入->右列神经元"与"输出->左列神经元"
的连线在几何上会直接穿过同一行相邻神经元的圆内（实测距圆心仅 0.05，半径 0.28），
即"线叠到一起"。这一版改用电路图常见的"总线（bus）+ 竖直支线"布局：
4 个神经元排成一行，输入侧用一条水平总线在行上方连接，每个神经元各有一条独立的竖直
支线接入总线；输出侧同理在行下方接一条总线。所有支线都是竖直线、各自 x 坐标不同，
彼此和其他神经元的圆之间永远不会有交叉。
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

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


def straight_arrow(ax, start, end, color="#333333", lw=1.3, ls="-"):
    a = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9,
                         linewidth=lw, color=color, linestyle=ls, clip_on=False)
    ax.add_patch(a)


row_y = 0.0
r = 0.30
neuron_xs = [2.6, 3.9, 5.2, 6.5]  # 4 个神经元排成一行，间距 1.3，直径 0.6，留有安全间隙
neuron_states = ["open", "closed", "open", "closed"]  # 对应正文 N1~N4
bus_top_y = row_y + r + 0.28   # 输入侧总线高度：留出足够间隙，明显高于圆顶
bus_bot_y = row_y - r - 0.28   # 输出侧总线高度：明显低于圆底

fig, ax = plt.subplots(figsize=(9.5, 1.6), dpi=170)

draw_box(ax, 0.6, row_y, 1.2, 0.5, "Input")
draw_box(ax, 8.5, row_y, 1.4, 0.5, "Output", facecolor="#E7F5EA", edgecolor="#2E7D46")

# 输入侧总线：从 Input 顶部升到 bus_top_y，横向拉到最后一个神经元，再各自竖直落到每个神经元顶部
ax.plot([0.6, 0.6], [row_y + 0.25, bus_top_y], color="#666666", lw=1.2, clip_on=False, zorder=1)
ax.plot([0.6, neuron_xs[-1]], [bus_top_y, bus_top_y], color="#666666", lw=1.2, clip_on=False, zorder=1)

# 输出侧总线：从最后一个神经元底部竖直落到 bus_bot_y，横向拉到 Output，再竖直接入 Output 顶部（含箭头）
ax.plot([neuron_xs[0], 8.5], [bus_bot_y, bus_bot_y], color="#666666", lw=1.2, clip_on=False, zorder=1)
straight_arrow(ax, (8.5, bus_bot_y), (8.5, row_y - 0.25), color="#666666")

for cx, state in zip(neuron_xs, neuron_states):
    if state == "open":
        fc, ec, ls_line, label = "#EAF2FB", "#1F5FA0", "-", "开"
    else:
        fc, ec, ls_line, label = "#F2F2F2", "#B33A3A", "--", "关"
    circ = Circle((cx, row_y), r, facecolor=fc, edgecolor=ec, linewidth=1.5, linestyle=ls_line, clip_on=False)
    ax.add_patch(circ)
    ax.text(cx, row_y, label, ha="center", va="center", fontsize=9,
             weight="bold", color=ec, clip_on=False)
    color = "#666666" if state == "open" else "#C9A0A0"
    ls = "-" if state == "open" else "--"
    # 竖直支线：从输入总线落到神经元顶部（含箭头）
    straight_arrow(ax, (cx, bus_top_y), (cx, row_y + r), color=color, ls=ls)
    # 竖直支线：从神经元底部接到输出总线
    ax.plot([cx, cx], [row_y - r, bus_bot_y], color=color, lw=1.2, ls=ls, clip_on=False, zorder=1)

ax.text(0.6, bus_top_y + 0.14, "扇出", fontsize=7.5, color="#666666", ha="center", clip_on=False)
ax.text(8.5, bus_bot_y - 0.14, "扇入", fontsize=7.5, color="#666666", ha="center", va="top", clip_on=False)

ax.set_xlim(-0.6, 9.6)
ax.set_ylim(bus_bot_y - 0.35, bus_top_y + 0.32)
ax.axis("off")
ax.set_title("Dropout：随机关闭部分神经元（● 实心=正常工作，○ 虚线=本次被关闭）",
             fontsize=9.3, weight="bold", pad=2)

output = Path(__file__).resolve().parent / "dropout_neurons_diagram.png"
fig.savefig(output, bbox_inches="tight")
print(f"saved to {output}")
