"""Draw the CoLA-SSL architecture figure -> paper/figs/fig_arch.pdf (vector)."""
import sys, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
out = sys.argv[1] if len(sys.argv) > 1 else "../paper/figs/fig_arch.pdf"
os.makedirs(os.path.dirname(out), exist_ok=True)
INK, TEAL, AMB, RED, BLUE, GREY = "#0B1F33", "#1B998B", "#F2A541", "#E4572E", "#3D5A80", "#6B7A8F"
plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(7.2, 2.55)); ax.set_xlim(0, 93); ax.set_ylim(6, 35); ax.axis("off")

def box(x, y, w, h, title, sub, color, tc="white", fs=7.2, ec=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=1.2", fc=color, ec=ec or "none", lw=1.2))
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=fs, fontweight="bold", color=tc)
    ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center", fontsize=fs - 1.6, color=tc)
def arrow(x0, y0, x1, y1, color=GREY, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, lw=1.2, color=color, linestyle=ls, shrinkA=0, shrinkB=0))

y, h = 16, 12
ax.text(6.0, y + h + 2.2, "Augmentation: training only", ha="left", fontsize=6.2, color=GREY, style="italic")
box(0.5, y, 10.5, h, "Waveform", "4 s, 16 kHz", BLUE)
box(12.8, y, 13.2, h, "Augment", "clean/noise/codec", BLUE)
ax.add_patch(FancyBboxPatch((28.5, y - 2.5), 17.0, h + 5, boxstyle="round,pad=0.0,rounding_size=1.2", fc=INK, ec="none"))
ax.text(37.0, y + h + 0.8, "Frozen WavLM", ha="center", va="center", fontsize=7.2, fontweight="bold", color="white")
for i in range(13):
    ax.add_patch(Rectangle((30.5 + i * 1.0, y - 0.8), 0.7, 5.2 + 0.25 * (i % 5), fc=AMB if i == 0 else TEAL, ec="none"))
ax.text(37.0, y - 1.7, "hidden states $h^{(0)}\\ldots h^{(12)}$", ha="center", va="center", fontsize=5.8, color="#CFE3F0")
box(48.0, y, 12.5, h, "Layer fusion", "$z_t=\\sum_l\\alpha_l h_t^{(l)}$", TEAL)
box(63.0, y, 13.0, h, "Attentive pool", "mean + std  \u2192  $f$", TEAL)
box(79.0, y + 6.5, 13.0, 8.5, "Classifier", "bona fide / spoof", AMB, tc=INK)
box(79.0, y - 4.5, 13.0, 8.5, "Domain head", "clean/noise/codec", "white", tc=RED, ec=RED)
arrow(11.0, y + h / 2, 12.8, y + h / 2); arrow(26.0, y + h / 2, 28.5, y + h / 2); arrow(45.5, y + h / 2, 48.0, y + h / 2)
arrow(60.5, y + h / 2, 63.0, y + h / 2)
arrow(76.0, y + h / 2 + 1.0, 79.0, y + 10.7)
arrow(76.0, y + h / 2 - 1.0, 79.0, y - 0.3, color=RED, ls=(0, (3, 2)))
ax.text(75.6, y + 1.6, "GRL", fontsize=6.3, color=RED, fontweight="bold", ha="right")
ax.text(54.2, y + h + 1.6, "learned $\\alpha$ (softmax)", ha="center", fontsize=6, color=TEAL)
ax.text(69.5, y + h + 1.6, "attention over time", ha="center", fontsize=6, color=TEAL)
ax.text(85.5, y + 17.4, "$\\mathcal{L}_{cls}$: weighted CE", ha="center", fontsize=6.3, color=INK)
ax.text(85.5, y - 6.6, "$\\lambda_{adv}\\,\\mathcal{L}_{dom}$, reversed grad.", ha="center", fontsize=6.3, color=RED)
ax.plot([48.0, 74.0], [y - 1.5, y - 1.5], color=GREY, lw=0.8)
ax.text(62.0, y - 3.6, "trained: fusion + pooling + heads (~1M params)", ha="center", fontsize=6.2, color=GREY, style="italic")
plt.tight_layout(pad=0.2); plt.savefig(out, bbox_inches="tight"); plt.savefig(out.replace(".pdf", ".png"), dpi=140, bbox_inches="tight"); print("saved", out)
