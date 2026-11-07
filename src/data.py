"""Dataset loaders: ASVspoof 2019 LA (train/dev/eval) and In-the-Wild (test only).
Label: 1 = bona fide, 0 = spoof.
"""
import os, random
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset
from aug import augment

SR, NSAMP = 16000, 64600  # ~4.04 s


def la_protocol(root, part):
    name = {"train": "train.trn", "dev": "dev.trl", "eval": "eval.trl"}[part]
    p = os.path.join(root, "ASVspoof2019_LA_cm_protocols", f"ASVspoof2019.LA.cm.{name}.txt")
    df = pd.read_csv(p, sep=" ", header=None, names=["spk", "utt", "env", "attack", "label"])
    df["path"] = [os.path.join(root, f"ASVspoof2019_LA_{part}", "flac", u + ".flac") for u in df.utt]
    df["y"] = (df.label == "bonafide").astype(int)
    return df


def itw_protocol(root):
    df = pd.read_csv(os.path.join(root, "release_in_the_wild", "meta.csv"))
    df.columns = [c.lower() for c in df.columns]
    df["path"] = [os.path.join(root, "release_in_the_wild", f) for f in df["file"]]
    df["y"] = df["label"].str.lower().isin(["bona-fide", "bonafide"]).astype(int)
    df["attack"] = np.where(df.y == 1, "-", "itw_spoof")
    return df


def fix_len(x, n, train):
    if len(x) >= n:
        s = random.randint(0, len(x) - n) if train else 0
        return x[s:s + n]
    return np.tile(x, int(np.ceil(n / len(x))))[:n]


class AudioSet(Dataset):
    def __init__(self, df, train=False, use_aug=False, nsamp=NSAMP):
        self.df, self.train, self.use_aug, self.n = df.reset_index(drop=True), train, use_aug, nsamp

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        x, sr = sf.read(r.path, dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        if sr != SR:
            from scipy.signal import resample_poly
            x = resample_poly(x, SR, sr).astype(np.float32)
        x = fix_len(x, self.n, self.train)
        dom = 0
        if self.train and self.use_aug:
            x, dom = augment(x)
        return torch.tensor(x, dtype=torch.float32), int(r.y), dom, i
