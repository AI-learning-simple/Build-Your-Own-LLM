#!/usr/bin/env python3
"""生成第10章 Scaled Dot-Product 一节的数形结合配图。
左图：点积标准差随维度 d_k 增长（数值验证 std = sqrt(d_k)）；
右图：四个 Key 与 Query 保持相同的、真实可控的余弦相似度（0.15/0.10/0.05/-0.05，
差异很小，属于常见的"相关但不悬殊"情形），在 d_k = 64 时对比缩放前后的分数与
Softmax 输出——直观展示"不缩放 -> 微小差异被放大成近似 one-hot"。
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

FONT_CANDIDATES = [
    "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC",
    "WenQuanYi Micro Hei", "Source Han Sans SC", "Arial Unicode MS",
]
available = {font.name for font in fm.fontManager.ttflist}
font = next((name for name in FONT_CANDIDATES if name in available), "DejaVu Sans")
plt.rcParams.update({"font.family": font, "axes.unicode_minus": False})

rng = np.random.default_rng(7)

# ---------- 左图：点积标准差随维度 d_k 增长 ----------
d_ks = [2, 8, 32, 128, 512]
n_trials = 4000
empirical_std = []
for d in d_ks:
    q = rng.normal(0, 1, size=(n_trials, d))
    k = rng.normal(0, 1, size=(n_trials, d))
    dots = np.sum(q * k, axis=1)
    empirical_std.append(dots.std())
theoretical_std = [np.sqrt(d) for d in d_ks]

# ---------- 右图：四个 Key，与 Query 的余弦相似度分别精确控制为下列数值 ----------
d_k = 64
cos_sims = [0.15, 0.10, 0.05, -0.05]
labels = [f"Key {c}\n(cos={cs:+.2f})" for c, cs in zip("ABCD", cos_sims)]

q_demo = rng.normal(0, 1, size=d_k)
q_norm = np.linalg.norm(q_demo)
q_dir = q_demo / q_norm

def make_key_with_cosine(q_dir, cos_sim, norm, rng):
    r = rng.normal(0, 1, size=q_dir.shape[0])
    r_orth = r - (r @ q_dir) * q_dir
    r_orth = r_orth / np.linalg.norm(r_orth)
    direction = cos_sim * q_dir + np.sqrt(max(1 - cos_sim ** 2, 0)) * r_orth
    return direction * norm

keys = [make_key_with_cosine(q_dir, cs, q_norm, rng) for cs in cos_sims]
raw_scores = np.array([q_demo @ k for k in keys])
scaled_scores = raw_scores / np.sqrt(d_k)

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

softmax_raw = softmax(raw_scores)
softmax_scaled = softmax(scaled_scores)

print("cos_sims =", cos_sims)
print("raw_scores =", np.round(raw_scores, 2))
print("scaled_scores =", np.round(scaled_scores, 2))
print("softmax_raw =", np.round(softmax_raw, 4))
print("softmax_scaled =", np.round(softmax_scaled, 4))

fig = plt.figure(figsize=(14.5, 5.6), dpi=170, constrained_layout=True)
grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.3], height_ratios=[1, 1])

# --- 左图：跨越整个左侧两行 ---
ax_std = fig.add_subplot(grid[:, 0])
ax_std.plot(d_ks, theoretical_std, "--", color="#999999", linewidth=1.8, label=r"理论值 $\sqrt{d_k}$", zorder=1)
ax_std.plot(d_ks, empirical_std, "o-", color="#B33A3A", linewidth=2.2, markersize=7,
            label="实测标准差（4000 次随机采样）", zorder=2)
for d, s in zip(d_ks, empirical_std):
    ax_std.annotate(f"{s:.1f}", (d, s), textcoords="offset points", xytext=(6, 8), fontsize=9, color="#B33A3A")
ax_std.set_xscale("log")
ax_std.set_xticks(d_ks, [str(d) for d in d_ks])
ax_std.set_xlabel(r"向量维度 $d_k$（对数坐标）", fontsize=10.5)
ax_std.set_ylabel("点积 q·k 的标准差", fontsize=10.5)
ax_std.set_title("维度越高，点积的波动范围越大\n（标准差按 √d_k 增长）", fontsize=12.5, weight="bold")
ax_std.legend(loc="upper left", fontsize=9.3)
ax_std.grid(alpha=0.3, which="both")

# --- 右上：缩放前后的分数对比 ---
ax_score = fig.add_subplot(grid[0, 1])
x = np.arange(len(labels))
width = 0.35
ax_score.bar(x - width / 2, raw_scores, width, label="原始分数 q·k（未缩放）", color="#B36B00", alpha=0.88)
ax_score.bar(x + width / 2, scaled_scores, width, label=r"缩放后 q·k / $\sqrt{d_k}$", color="#1F5FA0", alpha=0.88)
ax_score.axhline(0, color="#999999", linewidth=0.8)
ax_score.set_xticks(x, labels, fontsize=8.6)
ax_score.set_ylabel("分数", fontsize=10)
ax_score.set_title(f"四个 Key 与 Query 的余弦相似度只差 0.05~0.10，\nd_k={d_k} 时未缩放的分数却被放大到 ±10 量级", fontsize=11, weight="bold")
ax_score.legend(loc="upper right", fontsize=8.3)
ax_score.grid(axis="y", alpha=0.25)

# --- 右下：对应的 Softmax 输出对比 ---
ax_soft = fig.add_subplot(grid[1, 1])
ax_soft.bar(x - width / 2, softmax_raw, width, label="Softmax(原始分数) —— 接近 one-hot", color="#B36B00", alpha=0.88)
ax_soft.bar(x + width / 2, softmax_scaled, width, label="Softmax(缩放后分数) —— 分布更平滑", color="#1F5FA0", alpha=0.88)
ax_soft.set_xticks(x, labels, fontsize=8.6)
ax_soft.set_ylim(0, 1.05)
ax_soft.set_ylabel("Softmax 权重", fontsize=10)
ax_soft.set_title("不缩放：几乎所有权重压给了相似度最高的 Key（饱和、梯度趋近 0）\n缩放后：四个 Key 的权重差距回到合理范围，梯度正常流动", fontsize=10.8, weight="bold")
for xi, (r, s) in enumerate(zip(softmax_raw, softmax_scaled)):
    ax_soft.text(xi - width / 2, r + 0.02, f"{r:.2f}", ha="center", fontsize=8, color="#7A4A00")
    ax_soft.text(xi + width / 2, s + 0.02, f"{s:.2f}", ha="center", fontsize=8, color="#123A63")
ax_soft.legend(loc="upper right", fontsize=8.3)
ax_soft.grid(axis="y", alpha=0.25)

fig.suptitle("为什么要除以 √d_k：维度越高，点积越大，Softmax 越容易饱和", fontsize=15, weight="bold")
output = Path(__file__).resolve().parent / "scaled_dot_product_scaling.png"
fig.savefig(output, bbox_inches="tight")
print(f"saved to {output}")
