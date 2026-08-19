#!/usr/bin/env python3
"""
生成一张 3D 损失曲面 + 梯度方向示意图，风格参考吴恩达课程里的经典梯度下降图。
用于 6.5 节，帮助直观理解 "梯度" 这个概念：
- 曲面 = Loss 关于两个参数(w1, w2)的函数
- 曲面上某一点的"坡度最陡的上升方向" = 梯度 ∇L
- 梯度下降走的是它的反方向 -∇L（最陡下降方向）
- 一串小球轨迹 = 沿着负梯度方向、一步步走到碗底(Loss最小值)的过程
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.font_manager as fm

# ---- 中文字体：按常见优先级依次尝试 macOS / Windows / Ubuntu(Linux) 上可能安装的字体 ----
_CJK_FONT_CANDIDATES = [
    "PingFang SC", "Heiti SC", "STHeiti", "Arial Unicode MS",              # macOS
    "Microsoft YaHei", "Microsoft YaHei UI", "SimHei", "SimSun", "DengXian",  # Windows
    "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK TC",             # Ubuntu/Linux（常需 apt install fonts-noto-cjk）
    "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "Droid Sans Fallback",      # Ubuntu/Linux 备选
    "Source Han Sans SC", "Source Han Sans CN",                            # 思源黑体（跨平台）
]

_available_fonts = {f.name for f in fm.fontManager.ttflist}
_chosen_font = next((n for n in _CJK_FONT_CANDIDATES if n in _available_fonts), None)

if _chosen_font:
    plt.rcParams["font.family"] = _chosen_font
else:
    plt.rcParams["font.family"] = "sans-serif"
    print(
        "[警告] 未检测到可用的中文字体，图中中文可能显示为方块/乱码。\n"
        "  - macOS：一般自带 PingFang SC，无需处理；\n"
        "  - Windows：一般自带 Microsoft YaHei/SimHei，无需处理；\n"
        "  - Ubuntu/Linux：建议执行 `sudo apt install fonts-noto-cjk` 后重试。"
    )
plt.rcParams["axes.unicode_minus"] = False

# ---- 1. 定义一个不对称的"碗形"Loss曲面，更贴近真实损失函数（各方向曲率不同）----
def loss_fn(w1, w2):
    return 1.3 * w1**2 + 0.5 * w2**2

def grad_fn(w1, w2):
    return np.array([2.6 * w1, 1.0 * w2])

w1 = np.linspace(-3, 3, 120)
w2 = np.linspace(-3, 3, 120)
W1, W2 = np.meshgrid(w1, w2)
L = loss_fn(W1, W2)

fig = plt.figure(figsize=(11, 8.5), dpi=170)
ax = fig.add_subplot(111, projection="3d")

# 曲面
surf = ax.plot_surface(W1, W2, L, cmap=cm.viridis, alpha=0.55,
                        linewidth=0, antialiased=True, rstride=2, cstride=2)
ax.contour(W1, W2, L, zdir='z', offset=L.min() - 2, levels=12,
           cmap=cm.viridis, alpha=0.5)

# ---- 2. 当前点(随机初始化的位置) ----
cur = np.array([-2.3, 2.4])
cur_z = loss_fn(*cur)
ax.scatter([cur[0]], [cur[1]], [cur_z], color="red", s=90, zorder=10,
           label="当前参数位置 (w1, w2)")

# ---- 3. 梯度方向 ∇L (最陡上升方向) 与 负梯度方向 -∇L (下降方向) ----
g = grad_fn(*cur)
g_norm = g / np.linalg.norm(g)

arrow_len = 1.6
# 正梯度：红色箭头（指向 Loss 增大的方向）
ax.quiver(cur[0], cur[1], cur_z,
          g_norm[0]*arrow_len, g_norm[1]*arrow_len, 0,
          color="crimson", linewidth=3, arrow_length_ratio=0.18)
# 负梯度：绿色箭头（梯度下降真正走的方向，Loss减小最快）
ax.quiver(cur[0], cur[1], cur_z,
          -g_norm[0]*arrow_len, -g_norm[1]*arrow_len, 0,
          color="limegreen", linewidth=3, arrow_length_ratio=0.18)

ax.text(cur[0] + g_norm[0]*arrow_len*1.15, cur[1] + g_norm[1]*arrow_len*1.15, cur_z + 0.6,
        "grad L  (梯度：最陡上升方向)", color="crimson", fontsize=11, weight="bold")
ax.text(cur[0] - g_norm[0]*arrow_len*1.55, cur[1] - g_norm[1]*arrow_len*1.75, cur_z - 1.2,
        "-grad L (梯度下降走的方向)", color="green", fontsize=11, weight="bold")

# ---- 4. 沿负梯度做几步梯度下降，画出轨迹 ----
path = [cur.copy()]
p = cur.copy()
lr = 0.35
for _ in range(9):
    p = p - lr * grad_fn(*p)
    path.append(p.copy())
path = np.array(path)
path_z = loss_fn(path[:, 0], path[:, 1]) + 0.05

ax.plot(path[:, 0], path[:, 1], path_z, color="orange", marker="o",
        markersize=5, linewidth=2, zorder=9, label="梯度下降的路径 (6.6节)")
ax.scatter([0], [0], [loss_fn(0, 0)], color="gold", s=160, marker="*",
           edgecolor="black", zorder=11, label="Loss 最小值 (训练目标)")

ax.set_xlabel("参数 w1", labelpad=12)
ax.set_ylabel("参数 w2", labelpad=12)
ax.set_zlabel("Loss", labelpad=8)
ax.set_title("梯度 = Loss 曲面上某一点最陡上升的方向\n梯度下降 = 沿着它的反方向，一步步走向 Loss 最小值", fontsize=13, pad=18)
ax.view_init(elev=32, azim=-55)
ax.legend(loc="upper left", fontsize=9.5, framealpha=0.9)

fig.colorbar(surf, shrink=0.5, aspect=15, pad=0.08, label="Loss 大小")
plt.tight_layout()

OUTPUT_PATH = Path(__file__).resolve().parent / "gradient_descent_3d.png"
plt.savefig(OUTPUT_PATH, bbox_inches="tight")
print(f"saved to {OUTPUT_PATH}")
