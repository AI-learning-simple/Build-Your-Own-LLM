#!/usr/bin/env python3
"""生成第9章 Unrolling / Parameter Sharing 一节的 Hidden State 演变图。
用同一套 Wxh, Whh, b 处理 "I love AI today" 四个 Token，观察 Hidden State 如何逐步累积历史信息。
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

rng = np.random.default_rng(9)
tokens = ["I", "love", "AI", "today"]
embed_dim, hidden_size = 3, 4

embeddings = {
    "I": np.array([0.9, 0.1, -0.2]),
    "love": np.array([0.2, 0.8, 0.3]),
    "AI": np.array([-0.3, 0.4, 0.9]),
    "today": np.array([0.1, -0.5, 0.6]),
}
Wxh = rng.normal(0, 0.6, size=(hidden_size, embed_dim))
Whh = rng.normal(0, 0.5, size=(hidden_size, hidden_size))
b = rng.normal(0, 0.1, size=hidden_size)

h = np.zeros(hidden_size)
history = [h.copy()]
for tok in tokens:
    z = Wxh @ embeddings[tok] + Whh @ h + b
    h = np.tanh(z)
    history.append(h.copy())

H = np.array(history).T  # (hidden_size, T+1)

fig, (ax_heat, ax_line) = plt.subplots(1, 2, figsize=(13.5, 5.4), dpi=170,
                                        gridspec_kw={"width_ratios": [1, 1.35]},
                                        constrained_layout=True)

im = ax_heat.imshow(H, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax_heat.set_xticks(range(len(history)), ["h₀ = 0"] + [f"h{i+1}\n(读完 {t})" for i, t in enumerate(tokens)],
                    fontsize=9)
ax_heat.set_yticks(range(hidden_size), [f"dim {i+1}" for i in range(hidden_size)])
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        ax_heat.text(j, i, f"{H[i, j]:.2f}", ha="center", va="center", fontsize=8.3,
                     color="white" if abs(H[i, j]) > 0.55 else "black")
ax_heat.set_title("同一套 Wxh, Whh, b\n每一步都被反复调用（Parameter Sharing）", fontsize=11.8, weight="bold")
fig.colorbar(im, ax=ax_heat, shrink=0.85, label="Hidden State 数值 (∈ tanh 值域)")

colors = ["#1F5FA0", "#B36B00", "#3A8F5A", "#B33A3A"]
for d in range(hidden_size):
    ax_line.plot(range(len(history)), H[d], marker="o", color=colors[d], label=f"dim {d+1}", linewidth=2)
ax_line.axhline(1, color="#999999", linestyle="--", linewidth=0.8)
ax_line.axhline(-1, color="#999999", linestyle="--", linewidth=0.8)
ax_line.set_xticks(range(len(history)), ["h₀"] + [f"h{i+1}" for i in range(len(tokens))])
ax_line.set_xlabel("Time Step（读到第几个 Token）", fontsize=10.5)
ax_line.set_ylabel("Hidden State 各维数值", fontsize=10.5)
ax_line.set_ylim(-1.15, 1.15)
for i, tok in enumerate(tokens):
    ax_line.annotate(tok, (i + 1, 1.08), ha="center", fontsize=9.5, color="#555555")
ax_line.set_title("Hidden State 随 Token 不断更新\n——每一步都带着之前所有步的印记", fontsize=11.8, weight="bold")
ax_line.legend(loc="lower left", fontsize=9)
ax_line.grid(alpha=0.25)

fig.suptitle('Unrolling 数值示例：句子 "I love AI today" 的 Hidden State 演变', fontsize=14.5, weight="bold")
output = Path(__file__).resolve().parent / "rnn_hidden_state_evolution.png"
fig.savefig(output, bbox_inches="tight")
print(f"saved to {output}")
