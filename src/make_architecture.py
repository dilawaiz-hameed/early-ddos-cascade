"""Figure 0: implemented pipeline (boxes and arrows)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(10.5, 4.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")
INK, BLUE, GREEN, ORANGE, GREY = "#0b0b0b", "#2a78d6", "#1baf7a", "#eb6834", "#52514e"


def box(x, y, w, h, title, body, col):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc="white", ec=col, lw=1.4))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=8, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 5.6, body, ha="center", va="top", fontsize=6.4, color=GREY, linespacing=1.3)


def arr(x1, y1, x2, y2, col=GREY):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.1))


box(0.5, 26, 17, 18, "1  Packet captures", "CIC-DDoS2019 PCAPs\nexternal: StopDDoS +\nSanders + Malcolm\n(pcapio.py)", BLUE)
box(21, 26, 17, 18, "2  Flows + labels", "bidirectional 5-tuple\n60 s idle timeout\nIP-based labels\nfirst 32 packets kept", BLUE)
box(41.5, 26, 17, 18, "3  Early windows", "window (N, T) closes at\nN-th packet or T seconds\n31 features, 4 groups\n(features.py)", BLUE)
box(62, 26, 17, 18, "4  Two-stage cascade", "Stage 1: small DT\nat (N1, T1), confident\nflows decided here\nStage 2: light model\nat (N2, T2) (models.py)", GREEN)
box(82.5, 26, 17, 18, "5  Evaluation", "macro-F1, MCC, PR-AUC\nprecision at 1/10/30%\ntime-to-detect\nlatency, size, throughput", ORANGE)
box(41.5, 1, 37.5, 19, "NSGA-II search (train/validation only)", "genome: cascade on/off, windows, depth, model family,\nhyper-parameters, gate thresholds, feature mask\nf1 = 1 - macro-F1;  f2 = mean decision time\nf3 = normalised latency + size + CPU\nPareto front -> edge / enterprise / cloud profiles", GREEN)
box(0.5, 1, 37.5, 19, "Chronological split per capture", "60 / 20 / 20 by flow start time per capture\n(test is always later than train)\n10 time blocks: block bootstrap, Wilcoxon + Holm\ncross-dataset: CIC -> external and reverse", BLUE)
for x in (17.9, 38.4, 58.9, 79.4):
    arr(x, 35, x + 2.9, 35)
arr(60.25, 20.6, 68, 25.6, GREEN)
arr(19.2, 20.6, 29.5, 25.6, BLUE)
for ext in ("png", "pdf"):
    fig.savefig(f"figures/fig0_architecture.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
