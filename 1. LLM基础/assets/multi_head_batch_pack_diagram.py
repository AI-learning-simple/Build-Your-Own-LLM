#!/usr/bin/env python3
"""生成第 10 章多头注意力形状演变图（三维魔方风格，参考李沐《动手学深度学习 v2》）。

(B, T, d_model) --沿特征维切--> (B, T, H, d_head) --transpose 提 H 到前-->
(B, H, T, d_head) --折叠 B 与 H--> (B·H, T, d_head)

轴侧投影绘制长方体：X = 特征维、Y = Token 维 T、Z = 批次 / 头维。
左侧给出三轴图例，每个立方体在对应棱边标注维度，切分处画红色虚线切割标记，
最后一步用虚线框标出"折叠后的 B·H 份"。
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyArrowPatch, Polygon

FONT_CANDIDATES = [
    "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC",
    "WenQuanYi Micro Hei", "Source Han Sans SC", "Arial Unicode MS",
]
available = {font.name for font in fm.fontManager.ttflist}
font = next((name for name in FONT_CANDIDATES if name in available), "DejaVu Sans")
plt.rcParams.update({"font.family": font, "axes.unicode_minus": False})

# ===== 轴侧投影基向量 =====
EX = np.array([1.0, -0.34])   # X：特征维（d_model / d_head）
EY = np.array([0.60, 0.35])   # Y：Token 维 T
EZ = np.array([0.0, 1.0])     # Z：批次 / 头维

BLUE, BLUE_E = "#4C78A8", "#16324a"
ORANGE, ORANGE_E = "#F58518", "#8A4A0B"
GREY, GREY_E = "#B8C4CC", "#555555"
HEAD_F = [BLUE, ORANGE]
HEAD_E = [BLUE_E, ORANGE_E]
CUT = "#C0392B"

# 可视化尺寸
B, H = 2, 2
WX = 1.0      # 每个头在特征维占的宽度
DY = 1.4      # T 维深度
HZ = 0.45     # 每片在 Z 方向的高度
GAP_X = 0.16  # 切块之间的间隙
GAP_Z_IN = 0.07
GAP_Z_GRP = 0.26
STAGE_GAP = 1.7


def shade(color, factor):
    r, g, b = to_rgb(color)
    if factor >= 1:
        t = min(factor - 1, 1.0)
        return (r + (1 - r) * t, g + (1 - g) * t, b + (1 - b) * t)
    return (r * factor, g * factor, b * factor)


def project(origin, size, i, j, k):
    """轴侧投影：把长方体上的参数点 (i,j,k)∈[0,1]³ 投到 2D。"""
    ox, oy, oz = origin
    sx, sy, sz = size
    return (ox + i * sx) * EX + (oy + j * sy) * EY + (oz + k * sz) * EZ


def cube_faces(origin, size):
    P = lambda i, j, k: project(origin, size, i, j, k)
    front = [P(0, 0, 0), P(1, 0, 0), P(1, 0, 1), P(0, 0, 1)]
    right = [P(1, 0, 0), P(1, 1, 0), P(1, 1, 1), P(1, 0, 1)]
    top = [P(0, 0, 1), P(1, 0, 1), P(1, 1, 1), P(0, 1, 1)]
    return front, right, top


def draw_block(ax, origin, size, color, edge, lw=1.0):
    for poly, factor in zip(cube_faces(origin, size), (0.80, 0.90, 1.0)):
        ax.add_patch(Polygon(poly, closed=True, facecolor=shade(color, factor),
                             edgecolor=edge, lw=lw, clip_on=False))


def edge_mid(origin, size, a, b):
    return (project(origin, size, *a) + project(origin, size, *b)) / 2


def draw_axes_legend(ax, x, y):
    """三轴图例：X / Y / Z 分别是谁。"""
    L = 0.70
    specs = [(EX, "X：特征维", CUT, "left", "top"),
             (EY, "Y：Token 维 T", "#2E7D46", "left", "bottom"),
             (EZ, "Z：批次 / 头维", "#1F5FA0", "center", "bottom")]
    for vec, label, color, ha, va in specs:
        v = vec / np.linalg.norm(vec) * L
        ax.add_patch(FancyArrowPatch((x, y), (x + v[0], y + v[1]),
                                     arrowstyle="-|>", mutation_scale=13,
                                     lw=1.8, color=color, clip_on=False))
        ax.text(x + v[0] * 1.12, y + v[1] * 1.12, label, ha=ha, va=va,
                fontsize=8.5, color=color, weight="bold", clip_on=False)
    ax.text(x, y - 0.95, "三个方向\n分别是谁", ha="center", va="top",
            fontsize=8, color="#555555", clip_on=False)


fig, ax = plt.subplots(figsize=(15.5, 5.6), dpi=170)

# ===== 四个 stage 的块布局 =====
x1 = 2.4
x2 = x1 + (H * WX + DY * EY[0]) + STAGE_GAP
x3 = x2 + (H * WX + GAP_X + DY * EY[0]) + STAGE_GAP
x4 = x3 + (WX + DY * EY[0]) + STAGE_GAP

# stage1：B 片整块沿 Z 叠放，特征维未切
stage1 = [((x1 + 0.0, 0.0, b * (HZ + GAP_Z_IN)), (H * WX, DY, HZ), GREY) for b in range(B)]

# stage2：每片沿特征维切成 H 份
stage2 = [((x2 + h * (WX + GAP_X), 0.0, b * (HZ + GAP_Z_IN)), (WX, DY, HZ), HEAD_F[h])
          for b in range(B) for h in range(H)]

# stage3：H 挪到 Z 方向，摞间留分组间隙
z3 = []
for b in range(B):
    base = b * (H * (HZ + GAP_Z_IN) + GAP_Z_GRP)
    for h in range(H):
        z3.append(base + h * (HZ + GAP_Z_IN))
stage3 = [((x3 + 0.0, 0.0, z), (WX, DY, HZ), HEAD_F[idx % H]) for idx, z in enumerate(z3)]

# stage4：等间距连成一柱
z4 = [idx * (HZ + GAP_Z_IN) for idx in range(B * H)]
stage4 = [((x4 + 0.0, 0.0, z), (WX, DY, HZ), HEAD_F[idx % H]) for idx, z in enumerate(z4)]

stages = [stage1, stage2, stage3, stage4]
titles = ["(B, T, d_model)", "(B, T, H, d_head)", "(B, H, T, d_head)", "(B·H, T, d_head)"]
# 每个 stage 三条棱边上的维度标注：(特征维, Token 维, 批次/头维)
edge_labels = [("d_model", "T", "B"), ("d_head", "T", "B"),
               ("d_head", "T", "B × H"), ("d_head", "T", "B·H")]

for blocks in stages:
    for (ox, oy, oz), (sx, sy, sz), color in sorted(blocks, key=lambda it: (it[0][2], it[0][0])):
        draw_block(ax, (ox, oy, oz), (sx, sy, sz), color,
                   HEAD_E[0] if color == BLUE else (HEAD_E[1] if color == ORANGE else GREY_E))

# ===== 切分标记：在 stage1 的块上画"沿特征维切"的红色虚线 =====
for b in range(B):
    origin = (x1 + 0.0, 0.0, b * (HZ + GAP_Z_IN))
    size = (H * WX, DY, HZ)
    for h in range(1, H):
        i = h / H
        ax.plot(*zip(project(origin, size, i, 0, 0), project(origin, size, i, 0, 1)),
                color=CUT, lw=1.6, ls="--", clip_on=False, zorder=5)
        ax.plot(*zip(project(origin, size, i, 0, 1), project(origin, size, i, 1, 1)),
                color=CUT, lw=1.6, ls="--", clip_on=False, zorder=5)

# ===== stage4：虚线框标出"折叠后的 B·H 份" =====
z_top = max(z4) + HZ
p_lo = project((x4, 0, 0), (WX, DY, HZ), 0, 0, 0)
p_hi = project((x4, 0, 0), (WX, DY, HZ), 0, 0, 1)
frame_x = [x4 - 0.18, x4 + WX + DY * EY[0] + 0.18]
frame_y = [p_lo[1] - 0.16, p_hi[1] + z_top + 0.10]
ax.plot([frame_x[0], frame_x[1]], [frame_y[1], frame_y[1]], color="#2E7D46", lw=1.3, ls=(0, (4, 3)), clip_on=False)
ax.plot([frame_x[0], frame_x[1]], [frame_y[0], frame_y[0]], color="#2E7D46", lw=1.3, ls=(0, (4, 3)), clip_on=False)
ax.plot([frame_x[0], frame_x[0]], [frame_y[0], frame_y[1]], color="#2E7D46", lw=1.3, ls=(0, (4, 3)), clip_on=False)
ax.plot([frame_x[1], frame_x[1]], [frame_y[0], frame_y[1]], color="#2E7D46", lw=1.3, ls=(0, (4, 3)), clip_on=False)
ax.text(frame_x[1] + 0.15, (frame_y[0] + frame_y[1]) / 2,
        "B·H 份独立的\n(T, d_head)", ha="left", va="center", fontsize=8.5,
        color="#2E7D46", weight="bold", clip_on=False)

# ===== 每个 stage：标题 + 三条棱边的维度标注 =====
stage_bottoms = []
for blocks, title, (lab_x, lab_y, lab_z) in zip(stages, titles, edge_labels):
    top_y, bottom_y = -np.inf, np.inf
    min_x, max_x = np.inf, -np.inf
    for (ox, oy, oz), (sx, sy, sz), _ in blocks:
        for pt in sum(cube_faces((ox, oy, oz), (sx, sy, sz)), []):
            top_y = max(top_y, pt[1])
            bottom_y = min(bottom_y, pt[1])
            min_x = min(min_x, pt[0])
            max_x = max(max_x, pt[0])
    ax.text((min_x + max_x) / 2, top_y + 0.34, title, ha="center", va="bottom",
            fontsize=11, weight="bold", color="#16324a", clip_on=False)
    stage_bottoms.append(bottom_y)

    # 棱边标注取"最底层、最右侧"的块：轴侧投影下 X 增大会向右下走，
    # 若取左边的块，标签会落进它右下方那个块的 front 面里
    bottom_block = min(blocks, key=lambda it: it[0][2])
    bottom_row = [b for b in blocks if abs(b[0][2] - bottom_block[0][2]) < 1e-9]
    # 特征维 / Token 维标在最右侧块；批次维标在最左侧块（否则会落进邻块的面里）
    origin, size, _ = max(bottom_row, key=lambda it: it[0][0])
    origin_z, size_z, _ = min(bottom_row, key=lambda it: it[0][0])
    m_x = edge_mid(origin, size, (0, 0, 0), (1, 0, 0))       # 特征维棱
    m_y = edge_mid(origin, size, (1, 0, 0), (1, 1, 0))       # Token 维棱
    # 批次/头维：标整摞的左侧中部
    zs = [b[0][2] for b in blocks]
    mid_z = (min(zs), max(zs) + HZ)
    m_z_lo = project((origin_z[0], 0, mid_z[0]), (0, 0, 0), 0, 0, 0)
    m_z_hi = project((origin_z[0], 0, mid_z[1]), (0, 0, 0), 0, 0, 0)
    m_z = (m_z_lo + m_z_hi) / 2

    ax.text(m_x[0], m_x[1] - 0.14, lab_x, ha="center", va="top", fontsize=9,
            color=CUT, weight="bold", clip_on=False)
    # right 面是斜的，标签要往该棱边的"外侧下方"放，否则会落进面内
    ax.text(m_y[0] + 0.30, m_y[1] - 0.16, lab_y, ha="left", va="top", fontsize=9,
            color="#2E7D46", weight="bold", clip_on=False)
    ax.text(m_z[0] - 0.16, m_z[1], lab_z, ha="right", va="center", fontsize=9,
            color="#1F5FA0", weight="bold", clip_on=False)

# ===== 三轴图例 =====
draw_axes_legend(ax, 0.35, max(stage_bottoms) + 1.0)


def big_arrow(p1, p2, label):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=20,
                                 lw=1.7, color="#333333", clip_on=False))


# ===== stage 之间的箭头 =====
mid_y = HZ * 2
arrow_specs = [
    (x1 + H * WX + DY * EY[0] + 0.10, mid_y, x2 - 0.30, mid_y, "沿特征维切"),
    (x2 + H * WX + GAP_X + DY * EY[0] + 0.10, mid_y, x3 - 0.30, mid_y, "transpose(1, 2)"),
    (x3 + WX + DY * EY[0] + 0.10, mid_y, x4 - 0.30, mid_y, "折叠 B 与 H"),
]
for ax1, ay1, ax2, ay2, label in arrow_specs:
    big_arrow((ax1, ay1), (ax2, ay2), label)
    # 标签放在两个 stage 之间的空档里，避开立方体与棱边标注
    ax.text((ax1 + ax2) / 2, ay1 + 0.18, label, ha="center", va="bottom",
            fontsize=9, color="#333333", weight="bold", clip_on=False)

# ===== 底部注释 =====
bottom_y = min(stage_bottoms) - 0.75
ax.text((x1 + x4 + WX) / 2, bottom_y,
        "魔方的三条边分别是：X 特征维、Y Token 维 T、Z 批次 / 头维；"
        "先把特征维切成 H 段（每段 d_head），再把头挪到批次维旁，最后把 B 与 H 折叠成一个维度。",
        ha="center", va="top", fontsize=8.5, color="#555555", clip_on=False)

# ===== 画布范围 =====
xs_all, ys_all = [], []
for blocks in stages:
    for (ox, oy, oz), (sx, sy, sz), _ in blocks:
        for pt in sum(cube_faces((ox, oy, oz), (sx, sy, sz)), []):
            xs_all.append(pt[0])
            ys_all.append(pt[1])
ax.set_xlim(min(xs_all) - 1.5, max(xs_all) + 1.9)
ax.set_ylim(bottom_y - 0.6, max(ys_all) + 1.9)
ax.axis("off")
ax.set_title("Multi-Head Attention 形状演变：把特征维切成 H 段，再把头折叠进批次",
             fontsize=12.5, weight="bold", pad=10)

output = Path(__file__).resolve().parent / "multi_head_batch_pack_diagram.png"
fig.savefig(output, bbox_inches="tight")
print(f"saved to {output}")
