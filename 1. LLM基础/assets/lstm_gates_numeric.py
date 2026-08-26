#!/usr/bin/env python3
"""生成第9章 LSTM 三个门的数值示例图，对应文中"Alice、教授、北京大学、天气很好"笔记本例子，
覆盖 Forget Gate、Input Gate + Candidate、Output Gate 三节的公式。
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

slots = ["Alice", "教授", "北京大学", "天气很好"]
C_prev = np.array([0.90, 0.85, 0.80, 0.75])          # 旧 Cell State：四条笔记都还记着
f_t = np.array([0.95, 0.92, 0.88, 0.05])              # Forget Gate：前三条几乎保留，"天气很好"几乎清空
i_t = np.array([0.10, 0.15, 0.05, 0.15])              # Input Gate：暂时没有重要新信息要写入
C_candidate = np.array([0.20, 0.10, 0.05, 0.10])      # 候选新信息普遍偏低，天气一维也没有被新内容替代
C_t = f_t * C_prev + i_t * C_candidate                # 新 Cell State

# 问题："Alice 在哪里工作？" -> Output Gate 让"北京大学"这一维主导输出
o_t = np.array([0.20, 0.15, 0.90, 0.05])
h_t = o_t * np.tanh(C_t)

fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.4), dpi=170, constrained_layout=True)
x = np.arange(len(slots))
width = 0.25

# ---------- 图1：Forget / Input Gate + 候选信息 ----------
ax = axes[0]
ax.bar(x - width, f_t, width, label="Forget Gate f_t", color="#B33A3A", alpha=0.85)
ax.bar(x, i_t, width, label="Input Gate i_t", color="#3A8F5A", alpha=0.85)
ax.bar(x + width, C_candidate, width, label="候选新信息 C̃_t", color="#B36B00", alpha=0.85)
ax.set_xticks(x, slots, fontsize=10)
ax.set_ylim(0, 1.05)
ax.axhline(1, color="#999999", linestyle="--", linewidth=0.8)
ax.set_ylabel("数值 ∈ (0, 1)（Sigmoid 输出）", fontsize=10)
ax.set_title("第十一/十二节：Forget + Input Gate\n都是 Sigmoid 给出的\u201c比例\u201d，不是硬开关", fontsize=11.6, weight="bold")
ax.legend(loc="upper center", fontsize=8.3)
ax.grid(axis="y", alpha=0.25)

# ---------- 图2：Cell State 更新前后 ----------
ax = axes[1]
ax.bar(x - width / 2, C_prev, width, label="C_{t-1}（旧 Cell）", color="#8C8C8C", alpha=0.85)
ax.bar(x + width / 2, C_t, width, label="C_t = f_t⊙C_{t-1}+i_t⊙C̃_t", color="#1F5FA0", alpha=0.9)
ax.set_xticks(x, slots, fontsize=10)
ax.set_ylim(0, 1.05)
for xi, (old, new) in enumerate(zip(C_prev, C_t)):
    ax.annotate("", xy=(xi + width / 2, new), xytext=(xi - width / 2, old),
                arrowprops=dict(arrowstyle="->", color="#555555", linewidth=1.1))
ax.set_ylabel("Cell State 数值", fontsize=10)
ax.set_title("第十二节：Cell State 更新\n\u201c天气很好\u201d没有重要新信息，被大幅遗忘", fontsize=11.6, weight="bold")
ax.legend(loc="upper center", fontsize=9)
ax.grid(axis="y", alpha=0.25)

# ---------- 图3：Output Gate -> Hidden State ----------
ax = axes[2]
tanh_C = np.tanh(C_t)
ax.bar(x - width, tanh_C, width, label="tanh(C_t)（压缩到 (-1,1)）", color="#6A3D9A", alpha=0.85)
ax.bar(x, o_t, width, label="Output Gate o_t", color="#D08A00", alpha=0.9)
ax.bar(x + width, h_t, width, label="h_t = o_t × tanh(C_t)", color="#1F5FA0", alpha=0.9)
ax.set_xticks(x, slots, fontsize=10)
ax.set_ylim(0, 1.05)
ax.set_ylabel("数值", fontsize=10)
ax.set_title('第十三节：问 "Alice 在哪工作？"\nOutput Gate 让"北京大学"主导 h_t', fontsize=11.6, weight="bold")
ax.legend(loc="upper center", fontsize=8.3)
ax.grid(axis="y", alpha=0.25)

fig.suptitle('LSTM 三个门的完整数值示例（笔记本例子："Alice、教授、北京大学、天气很好"）', fontsize=15, weight="bold")
output = Path(__file__).resolve().parent / "lstm_gates_numeric.png"
fig.savefig(output, bbox_inches="tight")
print(f"saved to {output}")
print("C_t =", C_t)
print("h_t =", h_t)
