#!/usr/bin/env python3
"""生成第11章 残差连接一节的数值图：
无残差时，"每层输出 = 0.9×输入"的玩具链条（正文示例的假设）让前向信息与反向梯度都按 0.9^n 指数衰减；
残差的恒等支路提供一条不衰减的直通路径：信息分量恒为 x，梯度分量恒为 1（来自 y = x + F(x) 的 1）。

修复记录：
1）原先用 ax.annotate + arrowprops 画标注，但对数坐标（log scale）下 annotate 的文字
   在实测中完全不渲染（相同代码在线性坐标正常）。因此改为：
   - 文字用 ax.text + axes fraction 坐标定位（不依赖数据坐标，不受对数坐标影响）；
   - 箭头用支持混合坐标的 FancyArrowPatch 手画：竖直段落在目标 x 上，水平段接在文字块边缘，
     水平/垂直正交，没有斜角。
2）图例原先在 center left，正好压在两条曲线上。经逐角落实测，只有左下角是真正空白的
   （曲线已降到底部），已挪到左下角。
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

FONT_CANDIDATES = [
    "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC",
    "WenQuanYi Micro Hei", "Source Han Sans SC", "Arial Unicode MS",
]
available = {font.name for font in fm.fontManager.ttflist}
font = next((name for name in FONT_CANDIDATES if name in available), "DejaVu Sans")
plt.rcParams.update({"font.family": font, "axes.unicode_minus": False})


def add_ortho_label(ax, text, tip_x, tip_y, text_frac=(0.60, 0.42), top_frac=0.38,
                    h_gap=0.0, color="#B33A3A", fontsize=9.5):
    """正交标注：文字用 axes fraction 定位；箭头 = 目标 x 上的竖直线 + 接文字块边缘的水平线。
    这样水平/垂直正交、无斜角，且不受对数坐标影响（对数坐标下 ax.annotate 的文字实测不渲染）。"""
    mixed = ax.get_xaxis_transform()  # x 用数据坐标，y 用 axes fraction（0=底, 1=顶）
    ax.text(text_frac[0], text_frac[1], text, ha="left", va="bottom",
            fontsize=fontsize, color=color, weight="bold", transform=ax.transAxes, clip_on=False)
    # 竖直段：从文字块上沿高度垂直落到被标注的数据点（tip）
    vert = FancyArrowPatch((tip_x, top_frac), (tip_x, tip_y), transform=mixed,
                            arrowstyle="-", linewidth=1.1, color=color, clip_on=False)
    ax.add_patch(vert)
    # 水平段：从文字块右边缘水平接到目标 x（留出一点间隙，避免和文字粘连）
    horiz = FancyArrowPatch((tip_x - h_gap, top_frac), (tip_x, top_frac), transform=mixed,
                             arrowstyle="-|>", mutation_scale=11, linewidth=1.1, color=color,
                             clip_on=False)
    ax.add_patch(horiz)


n = np.arange(0, 101)
decay = 0.9 ** n

fig, (ax_fwd, ax_bwd) = plt.subplots(1, 2, figsize=(13, 5.3), dpi=170, constrained_layout=True)

# 左图：前向——原始信息 x=10 传到第 n 层还剩多少
ax_fwd.plot(n, 10 * decay, color="#B33A3A", linewidth=2.2,
            label="无残差：每层输出 = 0.9×输入（正文玩具假设）")
ax_fwd.plot(n, np.full_like(n, 10.0), color="#3A8F5A", linewidth=2.4, linestyle="--",
            label="残差：恒等支路携带的原始信息")
ax_fwd.set_yscale("log")
ax_fwd.set_ylim(3e-4, 30)
add_ortho_label(ax_fwd, "第 100 层只剩\n10×0.9¹⁰⁰ ≈ 2.7×10⁻⁴",
                tip_x=100, tip_y=2.66e-4, text_frac=(0.60, 0.42), top_frac=0.38, h_gap=0.35)
ax_fwd.set_xlabel("穿过的层数", fontsize=10.3)
ax_fwd.set_ylabel("原始信息 x=10 的剩余幅度（对数坐标）", fontsize=10.5)
ax_fwd.set_title("前向：原始信息还能剩多少", fontsize=12.3, weight="bold")
ax_fwd.legend(loc="lower left", fontsize=9.3, framealpha=0.95)  # 左下角：实测唯一不压线的位置
ax_fwd.grid(alpha=0.3, which="both")

# 右图：反向——梯度传回第 1 层时还有多大
ax_bwd.plot(n, decay, color="#B33A3A", linewidth=2.2,
            label="无残差：每层梯度 ×0.9（连乘）")
ax_bwd.plot(n, np.ones_like(n, dtype=float), color="#3A8F5A", linewidth=2.4, linestyle="--",
            label="残差：恒等支路贡献的梯度分量 = 1")
ax_bwd.set_yscale("log")
ax_bwd.set_ylim(8e-6, 3)
add_ortho_label(ax_bwd, "第 1 层收到的梯度\n0.9¹⁰⁰ ≈ 2.7×10⁻⁵",
                tip_x=100, tip_y=2.66e-5, text_frac=(0.58, 0.42), top_frac=0.38, h_gap=0.35)
ax_bwd.set_xlabel("反向穿过的层数", fontsize=10.3)
ax_bwd.set_ylabel("梯度大小（对数坐标）", fontsize=10.5)
ax_bwd.set_title("反向：梯度会不会消失", fontsize=12.3, weight="bold")
ax_bwd.legend(loc="lower left", fontsize=9.3, framealpha=0.95)
ax_bwd.grid(alpha=0.3, which="both")

fig.suptitle("残差连接：没有捷径时信息与梯度都指数衰减，恒等支路是一条不衰减的高速公路",
             fontsize=14.5, weight="bold")
output = Path(__file__).resolve().parent / "residual_identity_path.png"
fig.savefig(output, bbox_inches="tight")
print(f"saved to {output}")
