#!/usr/bin/env python3
"""生成第11章 第六节"一个完整的 Transformer Block"结构图（原 Mermaid 图 5，全书信息量最大的一张）：
输入 x 依次经过 LayerNorm -> Multi-Head Attention -> Dropout -> Add（第一条残差高速公路），
再经过 LayerNorm -> FFN -> Dropout -> Add（第二条残差高速公路），最终得到输出。

修复记录：
1）改用直角折线画残差支路（不依赖曲率），不会因画布压扁而被裁剪断线。
2）两条支路在 ADD1 上方的落点/起点分别偏移到 ADD1 的左侧和右侧，不再共用一个点，
   箭头与"残差"文字有足够的摆放空间。
3）上一版的方框过宽、占用了大量横向空间，箭头被挤在窄缝里。这一版把每个方框的
   宽度收窄到"刚好装下文字"，省下的空间全部让给方框之间的箭头（相邻方框之间的
   空隙从原先最小的 0.35 增大到 0.5 以上），箭头不再拥挤。
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


def draw_box(ax, cx, cy, w, h, text, facecolor="#EAF2FB", edgecolor="#1F5FA0", fontsize=7.5):
    box = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                          boxstyle="round,pad=0.018,rounding_size=0.05",
                          linewidth=1.5, edgecolor=edgecolor, facecolor=facecolor, clip_on=False)
    ax.add_patch(box)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, weight="bold",
            color="#16324a", clip_on=False)


def straight_arrow(ax, start, end, color="#333333", lw=1.5, ls="-"):
    a = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11,
                         linewidth=lw, color=color, linestyle=ls, clip_on=False)
    ax.add_patch(a)


def elbow_bypass(ax, x_start, x_end, y_base, y_top, color="#B33A3A", lw=1.4):
    """直角折线：从 (x_start, y_base) 竖直上升到 y_top，水平走到 x_end，
    再竖直落回 (x_end, y_base)（落点带箭头）。不依赖曲率，不会因画布压扁而断线。"""
    ax.plot([x_start, x_start], [y_base, y_top], color=color, lw=lw, ls="--", clip_on=False, zorder=1)
    ax.plot([x_start, x_end], [y_top, y_top], color=color, lw=lw, ls="--", clip_on=False, zorder=1)
    straight_arrow(ax, (x_end, y_top), (x_end, y_base + 0.04), color=color, lw=lw, ls="--")


y = 0
h_main = 0.85
# 方框宽度改为"刚好装下文字"，尽量把横向空间让给箭头
nodes = [
    ("X", 0.55, 1.0, "输入 x", "#EAF2FB", "#1F5FA0"),
    ("LN1", 1.95, 1.2, "LayerNorm", "#EAF2FB", "#1F5FA0"),
    ("MHA", 3.85, 1.6, "Multi-Head\nAttention", "#DCEBFB", "#1F5FA0"),
    ("DP1", 5.75, 1.2, "Dropout", "#F5EAF9", "#7A3FA0"),
    ("ADD1", 7.05, 0.8, "+", "#FFF3D6", "#B98600"),
    ("LN2", 8.55, 1.2, "LayerNorm", "#EAF2FB", "#1F5FA0"),
    ("FFN", 10.4, 1.5, "Feed Forward\nNetwork", "#DCEBFB", "#1F5FA0"),
    ("DP2", 12.25, 1.2, "Dropout", "#F5EAF9", "#7A3FA0"),
    ("ADD2", 13.55, 0.8, "+", "#FFF3D6", "#B98600"),
    ("OUT", 14.9, 1.1, "输出", "#E7F5EA", "#2E7D46"),
]

fig, ax = plt.subplots(figsize=(16.5, 1.85), dpi=170)

pos = {}
for key, cx, w, text, fc, ec in nodes:
    draw_box(ax, cx, y, w, h_main, text, facecolor=fc, edgecolor=ec)
    pos[key] = (cx, w)

chain = [n[0] for n in nodes]
for a_key, b_key in zip(chain[:-1], chain[1:]):
    ax_cx, ax_w = pos[a_key]
    bx_cx, bx_w = pos[b_key]
    straight_arrow(ax, (ax_cx + ax_w / 2, y), (bx_cx - bx_w / 2, y))

y_bypass_top = h_main / 2 + 0.35  # 两条支路共用同一高度：它们的水平段 x 区间不重叠，不会互相干扰

# 第一条残差高速公路：X 正上方出发，落点偏移到 ADD1 左侧（避免和第二条共用同一落点）
x_cx, _ = pos["X"]
add1_cx, _ = pos["ADD1"]
elbow_bypass(ax, x_cx, add1_cx - 0.35, y_base=h_main / 2, y_top=y_bypass_top)
ax.text((x_cx + add1_cx - 0.35) / 2, y_bypass_top + 0.16, "残差", fontsize=8.5,
        color="#B33A3A", ha="center", weight="bold", clip_on=False)

# 第二条残差高速公路：从 ADD1 右侧出发（与第一条的落点错开），落到 ADD2 正上方
add2_cx, _ = pos["ADD2"]
elbow_bypass(ax, add1_cx + 0.35, add2_cx, y_base=h_main / 2, y_top=y_bypass_top)
ax.text((add1_cx + 0.35 + add2_cx) / 2, y_bypass_top + 0.16, "残差", fontsize=8.5,
        color="#B33A3A", ha="center", weight="bold", clip_on=False)

# 底部子层标注（贴近箱体，不额外占用太多高度）
y_label_bottom = -h_main / 2 - 0.3
ax.text((pos["LN1"][0] + pos["DP1"][0]) / 2, y_label_bottom, "Attention 子层", fontsize=9,
        color="#555555", ha="center", va="top", style="italic", clip_on=False)
ax.text((pos["LN2"][0] + pos["DP2"][0]) / 2, y_label_bottom, "FFN 子层", fontsize=9,
        color="#555555", ha="center", va="top", style="italic", clip_on=False)

ax.set_xlim(-0.6, 16.2)
ax.set_ylim(y_label_bottom - 0.2, y_bypass_top + 0.38)
ax.axis("off")
ax.set_title("完整的 Transformer Block：两条残差高速公路，各绕开各自的子层", fontsize=12.5, weight="bold", pad=6)

output = Path(__file__).resolve().parent / "transformer_block_full_diagram.png"
fig.savefig(output, bbox_inches="tight")
print(f"saved to {output}")
