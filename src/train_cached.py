"""Train/evaluate the head on cached SSL features. Writes the same results/*.json + *_scores.npz as train.py.
  python train_cached.py --name cola_wavlm --mode wsum --adv --use_aug --seed 1
"""
import argparse, json, os, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from model import Detector
from metrics import compute_eer, min_tdcf, load_asv_scores, bootstrap_ci

p = argparse.ArgumentParser()
p.add_argument("--name", required=True); p.add_argument("--feats", default="../feats")
p.add_argument("--mode", default="wsum", choices=["last", "wsum"])
p.add_argument("--uniform", action="store_true"); p.add_argument("--adv", action="store_true"); p.add_argument("--use_aug", action="store_true")
p.add_argument("--adv_weight", type=float, default=0.3)
p.add_argument("--epochs", type=int, default=30); p.add_argument("--bs", type=int, default=64)
p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--seed", type=int, default=1)
p.add_argument("--asv_scores", default="../data/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt")
p.add_argument("--out", default="../results")
a = p.parse_args()
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
torch.set_num_threads(os.cpu_count() or 4); os.makedirs(a.out, exist_ok=True)
cfg = json.load(open(f"{a.feats}/config.json"))
X = {s: np.load(f"{a.feats}/{s}.npy", mmap_mode="r") for s in ("train", "dev", "eval", "itw") if os.path.exists(f"{a.feats}/{s}.npy")}
Mt = {s: np.load(f"{a.feats}/{s}_meta.npz") for s in X}
tr_idx = np.arange(len(Mt["train"]["y"]))
if not a.use_aug: tr_idx = tr_idx[Mt["train"]["dom"] == 0]
ytr, dtr = Mt["train"]["y"], Mt["train"]["dom"]
model = Detector(None, a.mode, a.adv, 0, cached_dims=(cfg["D"], cfg["L"]), uniform=a.uniform)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
steps = a.epochs * int(np.ceil(len(tr_idx) / a.bs)); sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
cw = torch.tensor([0.5, 0.5])    # train subset is class-balanced, so no re-weighting
batch = lambda split, idx: torch.from_numpy(np.ascontiguousarray(X[split][idx])).float().permute(1, 0, 2, 3)  # L,B,T,D


@torch.no_grad()
def score(split):
    model.eval(); n = len(Mt[split]["y"]); out = np.zeros(n)
    for i in range(0, n, 128):
        idx = np.arange(i, min(n, i + 128)); lg = model.head(batch(split, idx))[0]
        out[idx] = (lg[:, 1] - lg[:, 0]).numpy()
    return out


eer = lambda split, s: compute_eer(s[Mt[split]["y"] == 1], s[Mt[split]["y"] == 0])[0] * 100
best, best_state, log, step = 1e9, None, [], 0
for ep in range(a.epochs):
    model.train(); perm = np.random.permutation(tr_idx); run = 0
    for i in range(0, len(perm), a.bs):
        idx = np.sort(perm[i:i + a.bs]); model.lam = 2 / (1 + np.exp(-10 * step / steps)) - 1
        lg, dm, _ = model.head(batch("train", idx))
        y = torch.from_numpy(ytr[idx]).long()
        loss = F.cross_entropy(lg, y, weight=cw)
        if dm is not None: loss = loss + a.adv_weight * F.cross_entropy(dm, torch.from_numpy(dtr[idx]).long())
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); sched.step()
        step += 1; run += loss.item()
    e = eer("dev", score("dev")); log.append(dict(epoch=ep, loss=run / max(1, len(perm) / a.bs), dev_eer=e))
    if ep % 5 == 0 or ep == a.epochs - 1: print(log[-1], flush=True)
    if e < best: best = e; best_state = {k: v.clone() for k, v in model.state_dict().items()}
model.load_state_dict(best_state)
res = dict(name=a.name, seed=a.seed, args=vars(a), log=log, dev_eer=best); scores = {}
for split, key in (("eval", "la_eval"), ("itw", "itw")):
    if split not in X: continue
    s = score(split); y = Mt[split]["y"]; scores[key] = s
    res[f"{key}_eer"] = eer(split, s)
    res[f"{key}_ci95"] = [c * 100 for c in bootstrap_ci(s[y == 1], s[y == 0], 500)]
if "la_eval" in scores and os.path.exists(a.asv_scores):
    y = Mt["eval"]["y"]; s = scores["la_eval"]; res["la_eval_mintdcf"] = min_tdcf(s[y == 1], s[y == 0], load_asv_scores(a.asv_scores))
res["layer_weights"] = model.layer_weights().detach().tolist()
print(json.dumps({k: v for k, v in res.items() if k not in ("args", "log", "layer_weights")}, indent=1))
tag = f"{a.name}_s{a.seed}"; json.dump(res, open(f"{a.out}/{tag}.json", "w"), indent=1)
np.savez(f"{a.out}/{tag}_scores.npz", **{f"{k}_score": v for k, v in scores.items()},
         **{f"{k}_y": Mt["eval" if k == "la_eval" else "itw"]["y"] for k in scores},
         **{f"{k}_attack": Mt["eval" if k == "la_eval" else "itw"]["attack"] for k in scores})
