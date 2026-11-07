"""Extract frozen-SSL features ONCE (CPU-friendly). Time axis is average-pooled to --tpool frames, fp16.
Train split also gets --aug_copies augmented copies of every clip (domain labels 1/2) so that augmentation
and the adversarial head can be studied without re-running the SSL model.
Eval/dev/ITW subsets are class-balanced random samples (EER is prior-invariant); disclose this in the paper.
"""
import argparse, json, os, random
import numpy as np, pandas as pd, soundfile as sf, torch
from tqdm import tqdm
from data import la_protocol, itw_protocol, fix_len, SR, NSAMP
from aug import augment
from model import SSLFront

p = argparse.ArgumentParser()
p.add_argument("--ssl", default="microsoft/wavlm-base-plus")
p.add_argument("--la_root", default="../data/LA"); p.add_argument("--itw_root", default="../data/in_the_wild")
p.add_argument("--out", default="../feats")
p.add_argument("--n_train", type=int, default=1500); p.add_argument("--n_dev", type=int, default=400)
p.add_argument("--n_eval", type=int, default=1000); p.add_argument("--n_itw", type=int, default=1000)
p.add_argument("--aug_copies", type=int, default=1); p.add_argument("--tpool", type=int, default=50)
p.add_argument("--bs", type=int, default=8); p.add_argument("--seed", type=int, default=0)
p.add_argument("--splits", default="train,dev,eval,itw")
a = p.parse_args()
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
torch.set_num_threads(os.cpu_count() or 4)
os.makedirs(a.out, exist_ok=True)
front = SSLFront(a.ssl, 0).eval()
L, D = front.n_layers, front.dim
json.dump(dict(ssl=a.ssl, L=L, D=D, tpool=a.tpool), open(f"{a.out}/config.json", "w"))


def bal(df, n, seed):
    k = n // 2; b, s = df[df.y == 1], df[df.y == 0]
    return pd.concat([b.sample(min(k, len(b)), random_state=seed), s.sample(min(n - k, len(s)), random_state=seed)]) \
             .sample(frac=1, random_state=seed).reset_index(drop=True)


def load(path, train):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1: x = x.mean(1)
    if sr != SR:
        from scipy.signal import resample_poly
        x = resample_poly(x, SR, sr).astype(np.float32)
    return fix_len(x, NSAMP, train)


@torch.no_grad()
def extract(wavs):
    hs = front(torch.tensor(np.stack(wavs), dtype=torch.float32))   # L,B,T,D
    _, B, T, _ = hs.shape
    h = hs.permute(1, 0, 3, 2).reshape(B, L * D, T)
    h = torch.nn.functional.adaptive_avg_pool1d(h, a.tpool).reshape(B, L, D, a.tpool).permute(0, 1, 3, 2)
    return h.half().numpy()


frames = {"train": la_protocol(a.la_root, "train"), "dev": la_protocol(a.la_root, "dev"),
          "eval": la_protocol(a.la_root, "eval"), "itw": itw_protocol(a.itw_root)}
nn_ = {"train": a.n_train, "dev": a.n_dev, "eval": a.n_eval, "itw": a.n_itw}
for split in a.splits.split(","):
    df = bal(frames[split], nn_[split], a.seed)
    train = split == "train"
    K = 1 + (a.aug_copies if train else 0)
    M = len(df) * K
    mm = np.lib.format.open_memmap(f"{a.out}/{split}.npy", mode="w+", dtype=np.float16, shape=(M, L, a.tpool, D))
    y, att, dom, src = [], [], [], []
    buf, i0 = [], 0

    def flush():
        global buf, i0
        if not buf: return
        mm[i0:i0 + len(buf)] = extract([b for b in buf]); i0 += len(buf); buf = []

    for r in tqdm(df.itertuples(), total=len(df), desc=split):
        x = load(r.path, train)
        rows = [(x, 0)] + ([augment(x) for _ in range(a.aug_copies)] if train else [])
        for w, d in rows:
            w = fix_len(np.asarray(w, dtype=np.float32), NSAMP, False)
            buf.append(w); y.append(r.y); att.append(r.attack); dom.append(d); src.append(r.Index)
            if len(buf) >= a.bs: flush()
    flush(); mm.flush()
    np.savez(f"{a.out}/{split}_meta.npz", y=np.array(y), attack=np.array(att).astype(str), dom=np.array(dom), src=np.array(src))
    print(split, "done:", M, "clips")
