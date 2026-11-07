"""EDA for both datasets: class balance, duration stats, speaker disjointness -> paper/figs/fig_eda.pdf + stdout"""
import sys, numpy as np, soundfile as sf, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from data import la_protocol, itw_protocol
la, itw, out = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "../paper/figs")
import os; os.makedirs(out, exist_ok=True)
dfs = {f"LA-{p}": la_protocol(la, p) for p in ("train", "dev", "eval")}; dfs["ITW"] = itw_protocol(itw)
fig, ax = plt.subplots(figsize=(3.4, 2.2))
for k, d in dfs.items():
    sub = d.sample(min(1500, len(d)), random_state=0)
    dur = np.array([sf.info(p).duration for p in sub.path])
    print(f"{k}: n={len(d)} bona={int(d.y.sum())} spoof={int((1-d.y).sum())} dur mean={dur.mean():.2f}s med={np.median(dur):.2f}s")
    ax.hist(dur, 40, range=(0, 15), histtype="step", density=True, label=k)
ax.set_xlabel("duration (s)"); ax.legend(fontsize=6); plt.tight_layout(); plt.savefig(f"{out}/fig_eda.pdf")
S = {k: set(d.spk) for k, d in dfs.items() if k.startswith("LA")}
print("speaker overlap train/dev, train/eval, dev/eval:", len(S["LA-train"] & S["LA-dev"]), len(S["LA-train"] & S["LA-eval"]), len(S["LA-dev"] & S["LA-eval"]))
