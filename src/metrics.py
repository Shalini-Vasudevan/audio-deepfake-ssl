"""EER and min t-DCF (ASVspoof 2019 LA definition, Kinnunen et al. 2018).
Convention everywhere: higher score = more bona fide.
"""
import numpy as np


def det_curve(bona, spoof):
    scores = np.concatenate([bona, spoof])
    labels = np.concatenate([np.ones_like(bona), np.zeros_like(spoof)])
    idx = np.argsort(scores, kind="mergesort")
    labels = labels[idx]
    tp = np.cumsum(labels)                 # bona rejected when thr just above
    fp = np.cumsum(1 - labels)
    p_miss = np.concatenate([[0.0], tp / len(bona)])         # bona rejected
    p_fa = np.concatenate([[1.0], 1 - fp / len(spoof)])      # spoof accepted
    thr = np.concatenate([[scores[idx][0] - 1e-3], scores[idx]])
    return p_miss, p_fa, thr


def compute_eer(bona, spoof):
    p_miss, p_fa, thr = det_curve(np.asarray(bona, float), np.asarray(spoof, float))
    i = np.nanargmin(np.abs(p_miss - p_fa))
    return float((p_miss[i] + p_fa[i]) / 2), float(thr[i])


COST = dict(Pspoof=0.05, Ptar=0.95 * 0.99, Pnon=0.95 * 0.01,
            Cmiss_asv=1, Cfa_asv=10, Cmiss_cm=1, Cfa_cm=10)


def asv_error_rates(asv_tar, asv_non, asv_spoof, thr):
    p_miss = np.mean(asv_tar < thr)
    p_fa = np.mean(asv_non >= thr)
    p_miss_spoof = np.mean(asv_spoof < thr)
    return p_miss, p_fa, p_miss_spoof


def load_asv_scores(path):
    """ASVspoof2019.LA.asv.eval.gi.trl.scores.txt: <source> <key> <score>; key in target/nontarget/spoof."""
    d = np.genfromtxt(path, dtype=str)
    keys, sc = d[:, 1], d[:, 2].astype(float)
    return sc[keys == "target"], sc[keys == "nontarget"], sc[keys == "spoof"]


def min_tdcf(bona_cm, spoof_cm, asv_scores):
    tar, non, spf = asv_scores
    _, thr = compute_eer(tar, non)                    # ASV operating point at its EER
    p_miss_asv, p_fa_asv, p_miss_spoof_asv = asv_error_rates(tar, non, spf, thr)
    c = COST
    C1 = c["Ptar"] * (c["Cmiss_cm"] - c["Cmiss_asv"] * p_miss_asv) - c["Pnon"] * c["Cfa_asv"] * p_fa_asv
    C2 = c["Cfa_cm"] * c["Pspoof"] * (1 - p_miss_spoof_asv)
    p_miss_cm, p_fa_cm, _ = det_curve(np.asarray(bona_cm, float), np.asarray(spoof_cm, float))
    tdcf = C1 * p_miss_cm + C2 * p_fa_cm
    return float(np.min(tdcf / np.minimum(C1, C2)))


def bootstrap_ci(bona, spoof, n=1000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    bona, spoof = np.asarray(bona), np.asarray(spoof)
    e = [compute_eer(rng.choice(bona, len(bona)), rng.choice(spoof, len(spoof)))[0] for _ in range(n)]
    return float(np.percentile(e, 100 * alpha / 2)), float(np.percentile(e, 100 * (1 - alpha / 2)))
