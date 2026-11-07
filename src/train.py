"""Train on ASVspoof2019 LA, select on LA dev, report on LA eval (in-domain) and In-the-Wild (cross-dataset).
Example:
  python train.py --name cola_wavlm --ssl microsoft/wavlm-large --mode wsum --aug --adv --seed 1
"""
import argparse, json, os, random, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import AudioSet, la_protocol, itw_protocol
from model import Detector
from metrics import compute_eer, min_tdcf, load_asv_scores, bootstrap_ci

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--la_root", default="../data/LA")
p.add_argument("--itw_root", default="../data/in_the_wild")
p.add_argument("--asv_scores", default="../data/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt")
p.add_argument("--ssl", default="microsoft/wavlm-large")
p.add_argument("--mode", default="wsum", choices=["last", "wsum", "lfcc"])
p.add_argument("--aug", action="store_true")
p.add_argument("--adv", action="store_true")
p.add_argument("--adv_weight", type=float, default=0.3)
p.add_argument("--finetune_top", type=int, default=0)
p.add_argument("--epochs", type=int, default=6)
p.add_argument("--bs", type=int, default=16)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--lr_ssl", type=float, default=1e-5)
p.add_argument("--workers", type=int, default=4)
p.add_argument("--max_train", type=int, default=0, help="subsample train set (0 = all)")
p.add_argument("--max_eval", type=int, default=0, help="subsample dev/eval/itw (debug)")
p.add_argument("--seed", type=int, default=1)
p.add_argument("--out", default="../results")
p.add_argument("--no_itw", action="store_true")
a = p.parse_args()

random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(a.out, exist_ok=True)
tag = f"{a.name}_s{a.seed}"

tr, dv, ev = (la_protocol(a.la_root, s) for s in ("train", "dev", "eval"))
# leakage check: speaker-disjoint partitions (official LA design); fail loudly if violated
assert not (set(tr.spk) & set(dv.spk)) and not (set(tr.spk) & set(ev.spk)) and not (set(dv.spk) & set(ev.spk)), "speaker leakage"
if a.max_train:
    tr = tr.sample(min(a.max_train, len(tr)), random_state=a.seed)
if a.max_eval:
    dv, ev = dv.sample(min(a.max_eval, len(dv)), random_state=0), ev.sample(min(a.max_eval, len(ev)), random_state=0)
sets = {"la_eval": ev}
if not a.no_itw:
    itw = itw_protocol(a.itw_root)
    sets["itw"] = itw.sample(min(a.max_eval, len(itw)), random_state=0) if a.max_eval else itw

mk = lambda df, train: DataLoader(AudioSet(df, train, a.aug), a.bs, shuffle=train, num_workers=a.workers,
                                  drop_last=train, pin_memory=True)
tl = mk(tr, True)

model = Detector(a.ssl, a.mode, a.adv, a.finetune_top).to(dev)
ssl_params = [q for n, q in model.named_parameters() if "front" in n and q.requires_grad]
oth = [q for n, q in model.named_parameters() if "front" not in n]
opt = torch.optim.AdamW([{"params": oth, "lr": a.lr}] + ([{"params": ssl_params, "lr": a.lr_ssl}] if ssl_params else []),
                        weight_decay=1e-4)
total = a.epochs * len(tl)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in opt.param_groups], total_steps=total, pct_start=max(0.1, 3 / total))
scaler = torch.cuda.amp.GradScaler(enabled=dev == "cuda")
cw = torch.tensor([0.1, 0.9], device=dev)  # [spoof, bona]; bona fide is the minority class


@torch.no_grad()
def score(df):
    model.eval()
    out = np.zeros(len(df))
    for x, y, d, i in tqdm(DataLoader(AudioSet(df, False, False), 32, num_workers=a.workers), leave=False):
        with torch.autocast(dev, enabled=dev == "cuda"):
            lg = model(x.to(dev))[0].float()
        out[i.numpy()] = (lg[:, 1] - lg[:, 0]).cpu().numpy()   # bona-fide log-odds
    return out


def eer_of(df, s):
    return compute_eer(s[df.y.values == 1], s[df.y.values == 0])[0] * 100


best, best_state, log, step = 1e9, None, [], 0
for ep in range(a.epochs):
    model.train(); t0 = time.time(); run = 0
    for x, y, d, _ in tqdm(tl, desc=f"ep{ep}", leave=False):
        model.lam = 2 / (1 + np.exp(-10 * step / total)) - 1   # DANN schedule 0 -> 1
        x, y, d = x.to(dev), y.to(dev), d.to(dev)
        with torch.autocast(dev, enabled=dev == "cuda"):
            lg, dm, _ = model(x)
            loss = F.cross_entropy(lg.float(), y, weight=cw)
            if dm is not None:
                loss = loss + a.adv_weight * F.cross_entropy(dm.float(), d)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt); scaler.update(); sched.step(); step += 1; run += loss.item()
    e = eer_of(dv, score(dv))
    log.append(dict(epoch=ep, loss=run / len(tl), dev_eer=e, sec=time.time() - t0))
    print(log[-1], flush=True)
    if e < best:
        best = e; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "front" not in k or ssl_params}

model.load_state_dict(best_state, strict=False)
res = dict(name=a.name, seed=a.seed, args=vars(a), log=log, dev_eer=best)
scores = {}
for k, df in sets.items():
    s = score(df); scores[k] = s
    res[f"{k}_eer"] = eer_of(df, s)
    res[f"{k}_ci95"] = [c * 100 for c in bootstrap_ci(s[df.y.values == 1], s[df.y.values == 0], 500)]
if "la_eval" in scores and os.path.exists(a.asv_scores):
    s = scores["la_eval"]; y = ev.y.values
    res["la_eval_mintdcf"] = min_tdcf(s[y == 1], s[y == 0], load_asv_scores(a.asv_scores))
if a.mode != "lfcc":
    res["layer_weights"] = model.layer_weights().detach().cpu().tolist()
print(json.dumps({k: v for k, v in res.items() if k not in ("args", "log", "layer_weights")}, indent=1))
json.dump(res, open(f"{a.out}/{tag}.json", "w"), indent=1)
np.savez(f"{a.out}/{tag}_scores.npz", **{f"{k}_score": v for k, v in scores.items()},
         **{f"{k}_y": sets[k].y.values for k in scores}, **{f"{k}_attack": sets[k].attack.values.astype(str) for k in scores})
