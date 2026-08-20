#!/usr/bin/env python3
"""生成 17.2 节从波形到离散 Code 的共享时间轴图。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

sr = 8000
t_a = np.arange(int(sr * 0.5)) / sr
t_b = np.arange(int(sr * 0.5)) / sr
seg_a = 0.5 * np.sin(2 * np.pi * 440 * t_a)
seg_s = np.zeros(int(sr * 0.2))
seg_b = 0.5 * np.sin(2 * np.pi * 880 * t_b)
audio = np.concatenate([seg_a, seg_s, seg_b])
audio += 0.01 * np.random.RandomState(0).randn(len(audio))
frame_len = 160
frames = np.array([audio[i:i + frame_len] for i in range(0, len(audio) - frame_len + 1, frame_len)])


def frame_features(frame):
    rms = np.sqrt(np.mean(frame ** 2))
    spectrum = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
    frequencies = np.fft.rfftfreq(len(frame), 1 / sr)
    peak = frequencies[np.argmax(spectrum)] if rms > 0.02 else 0.0
    zcr = np.mean(np.abs(np.diff(np.signbit(frame).astype(float))))
    return np.array([rms, peak, zcr])

X = np.array([frame_features(frame) for frame in frames])
mu, sd = X.mean(0), X.std(0)
Xn = (X - mu) / (sd + 1e-9)


def kmeans(data, K=8, iters=50, seed=0):
    rng = np.random.RandomState(seed)
    centers = data[rng.choice(len(data), K, replace=False)].copy()
    assignment = np.zeros(len(data), dtype=int)
    for _ in range(iters):
        distances = ((data[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        assignment = distances.argmin(1)
        for index in range(K):
            if np.any(assignment == index):
                centers[index] = data[assignment == index].mean(0)
    return assignment

codes = kmeans(Xn)
time = np.arange(len(audio)) / sr
frame_time = (np.arange(len(frames)) + 0.5) * frame_len / sr

fig, axes = plt.subplots(5, 1, figsize=(14, 9.5), dpi=170, sharex=True,
                         gridspec_kw={"height_ratios": [2.0, 1.0, 1.0, 1.0, 0.72]}, constrained_layout=True)
regions = [(0.0, 0.5, "440 Hz", "#DDEBF7"), (0.5, 0.7, "silence", "#EEEEEE"), (0.7, 1.2, "880 Hz", "#FCE4D6")]
for axis in axes:
    for left, right, _, color in regions:
        axis.axvspan(left, right, color=color, alpha=0.42, zorder=0)
    axis.axvline(0.5, color="#777777", linestyle="--", linewidth=0.8)
    axis.axvline(0.7, color="#777777", linestyle="--", linewidth=0.8)

axes[0].plot(time, audio, color="#333333", linewidth=0.45)
axes[0].set_ylabel("amplitude")
axes[0].set_title("Waveform: 9600 samples are divided into 60 non-overlapping 20 ms frames", weight="bold")
for boundary in np.arange(0, 1.201, 0.1):
    axes[0].axvline(boundary, color="#999999", linewidth=0.35, alpha=0.35)
for left, right, label, _ in regions:
    axes[0].text((left + right) / 2, 0.93, label, transform=axes[0].get_xaxis_transform(), ha="center", va="top", weight="bold")

axes[1].plot(frame_time, X[:, 0], "-o", markersize=2.8, color="#1F77B4")
axes[1].set_ylabel("RMS")
axes[1].grid(alpha=0.18)
axes[2].step(frame_time, X[:, 1], where="mid", color="#D95F02", linewidth=1.8)
axes[2].set_ylabel("peak Hz")
axes[2].set_yticks([0, 450, 900])
axes[2].grid(alpha=0.18)
axes[3].plot(frame_time, X[:, 2], "-o", markersize=2.8, color="#2E8B57")
axes[3].set_ylabel("zero-crossing rate")
axes[3].grid(alpha=0.18)

cmap = ListedColormap(["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#FF9DA6", "#9D755D"])
norm = BoundaryNorm(np.arange(-0.5, 8.5, 1), cmap.N)
axes[4].imshow(codes[None, :], aspect="auto", extent=[0, 1.2, 0, 1], cmap=cmap, norm=norm, interpolation="nearest")
axes[4].set_yticks([])
axes[4].set_ylabel("Code ID")
for index, code in enumerate(codes):
    if index % 2 == 0:
        axes[4].text((index + 0.5) * 0.02, 0.5, str(code), ha="center", va="center", fontsize=6.5, color="white")
axes[4].set_xlabel("time (seconds)")
axes[4].set_xlim(0, 1.2)

fig.suptitle("Audio Tokenization: waveform -> frames -> features -> discrete Code IDs", fontsize=15, weight="bold")
output = Path(__file__).resolve().parent / "audio_tokenization_timeline.png"
fig.savefig(output, bbox_inches="tight")
print(f"saved to {output}")
