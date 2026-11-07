"""Training-time augmentation: additive noise, RawBoost-style convolutive/non-linear distortion,
and codec/channel simulation (MP3, Opus via ffmpeg if present; mu-law 8 kHz telephony always).
Returns (waveform, domain_id): 0 clean, 1 noise/rawboost, 2 codec/channel.  domain_id feeds the
adversarial head in model.py.
"""
import io, os, random, shutil, subprocess, tempfile
import numpy as np
import soundfile as sf
from scipy.signal import firwin, lfilter, resample_poly

SR = 16000
HAS_FFMPEG = shutil.which("ffmpeg") is not None


def add_noise(x, snr_db=None):
    snr_db = snr_db if snr_db is not None else random.uniform(5, 30)
    n = np.random.randn(len(x))
    if random.random() < 0.5:                      # coloured (pink-ish) noise
        n = lfilter([1.0], [1.0, -random.uniform(0.5, 0.95)], n)
    px, pn = np.mean(x ** 2) + 1e-9, np.mean(n ** 2) + 1e-9
    return x + n * np.sqrt(px / (pn * 10 ** (snr_db / 10)))


def rawboost_lite(x):
    """Random band-limited FIR (linear convolutive noise) + soft non-linear distortion."""
    if random.random() < 0.7:
        lo = random.uniform(20, 300); hi = random.uniform(3000, 7800)
        h = firwin(random.choice([31, 63, 127]), [lo, hi], pass_zero=False, fs=SR)
        x = lfilter(h, [1.0], x)
    if random.random() < 0.5:
        g = random.uniform(1.0, 4.0)
        x = np.tanh(g * x / (np.max(np.abs(x)) + 1e-9)) / np.tanh(g) * np.max(np.abs(x))
    return x


def mulaw_telephony(x):
    y = resample_poly(x, 1, 2)                     # 16k -> 8k
    mu = 255.0
    m = np.max(np.abs(y)) + 1e-9
    c = np.sign(y / m) * np.log1p(mu * np.abs(y / m)) / np.log1p(mu)
    c = np.round((c + 1) / 2 * 255) / 255 * 2 - 1  # 8-bit quantisation
    y = np.sign(c) * ((1 + mu) ** np.abs(c) - 1) / mu * m
    return resample_poly(y, 2, 1)[: len(x)]


def ffmpeg_codec(x, kind):
    with tempfile.TemporaryDirectory() as d:
        src, mid, out = (os.path.join(d, f) for f in ("a.wav", "b." + ("mp3" if kind == "mp3" else "ogg"), "c.wav"))
        sf.write(src, x.astype(np.float32), SR)
        if kind == "mp3":
            enc = ["-codec:a", "libmp3lame", "-b:a", f"{random.choice([16, 24, 32, 48, 64])}k"]
        else:
            enc = ["-codec:a", "libopus", "-b:a", f"{random.choice([8, 12, 16, 24])}k"]
        q = ["ffmpeg", "-loglevel", "error", "-y", "-i"]
        subprocess.run(q + [src] + enc + [mid], check=True)
        subprocess.run(q + [mid, "-ar", str(SR), out], check=True)
        y, _ = sf.read(out)
    y = y[: len(x)]
    return np.pad(y, (0, max(0, len(x) - len(y))))


def codec(x):
    kinds = ["mulaw"] + (["mp3", "opus"] if HAS_FFMPEG else [])
    k = random.choice(kinds)
    try:
        return mulaw_telephony(x) if k == "mulaw" else ffmpeg_codec(x, "mp3" if k == "mp3" else "opus")
    except Exception:
        return mulaw_telephony(x)


def augment(x, p_clean=0.2):
    r = random.random()
    if r < p_clean:
        return x, 0
    if r < p_clean + (1 - p_clean) * 0.45:
        y = rawboost_lite(x)
        if random.random() < 0.6:
            y = add_noise(y)
        return y, 1
    y = codec(x)
    if random.random() < 0.3:
        y = add_noise(y)
    return y, 2
