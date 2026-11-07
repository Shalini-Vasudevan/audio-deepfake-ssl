"""Explainability / error analysis figures -> paper/figs/*.pdf"""
import glob, json, sys, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from metrics import compute_eer
rd = sys.argv[1] if len(sys.argv) > 1 else "../results"; od = sys.argv[2] if len(sys.argv) > 2 else "../paper/figs"
os.makedirs(od, exist_ok=True); plt.rcParams.update({"font.size": 8, "figure.figsize": (3.4, 2.3)})
def jl(n): return [json.load(open(f)) for f in sorted(glob.glob(f"{rd}/{n}_s*.json"))]
# 1) learned layer weights
plt.figure()
for n, lab in [("wavlm_wsum_noaug", "LAF"), ("cola_wavlm", "CoLA")]:
    rs = jl(n)
    if rs: w = np.mean([r["layer_weights"] for r in rs], 0); plt.plot(range(len(w)), w, marker="o", ms=2, label=lab)
plt.xlabel("WavLM-L layer (0 = CNN output)"); plt.ylabel("softmax weight"); plt.legend(); plt.tight_layout()
plt.savefig(f"{od}/fig_layers.pdf"); plt.close()
# 2) per-attack EER on LA eval
plt.figure(figsize=(3.4, 2.4)); width = 0.4
for j, (n, lab) in enumerate([("wavlm_last", "WavLM last"), ("cola_wavlm", "CoLA")]):
    fs = sorted(glob.glob(f"{rd}/{n}_s*_scores.npz"))
    if not fs: continue
    z = np.load(fs[0]); s, y, at = z["la_eval_score"], z["la_eval_y"], z["la_eval_attack"]
    atts = sorted(a for a in set(at) if a != "-")
    e = [compute_eer(s[y == 1], s[at == a])[0] * 100 for a in atts]
    plt.bar(np.arange(len(atts)) + j * width, e, width, label=lab)
    plt.xticks(np.arange(len(atts)) + width / 2, [a.replace("A", "") for a in atts], fontsize=6)
plt.xlabel("Attack ID (A07--A19)"); plt.ylabel("EER (%)"); plt.legend(); plt.tight_layout()
plt.savefig(f"{od}/fig_attack_eer.pdf"); plt.close()
# 3) ITW score histograms (calibration / error cases)
fig, ax = plt.subplots(1, 2, figsize=(3.4, 2.0), sharey=True)
for a_, (n, lab) in zip(ax, [("wavlm_last", "WavLM last"), ("cola_wavlm", "CoLA")]):
    fs = sorted(glob.glob(f"{rd}/{n}_s*_scores.npz"))
    if not fs: continue
    z = np.load(fs[0]); s, y = z["itw_score"], z["itw_y"]
    a_.hist(s[y == 1], 50, alpha=.6, label="bona fide", density=True); a_.hist(s[y == 0], 50, alpha=.6, label="spoof", density=True)
    a_.set_title(lab, fontsize=7); a_.set_xlabel("bona-fide log-odds")
ax[0].legend(fontsize=6); plt.tight_layout(); plt.savefig(f"{od}/fig_itw_scores.pdf"); plt.close()
print("figs written to", od)
