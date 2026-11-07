# Generalizable Audio Deepfake Detection: Cross-Dataset Spoofed-Speech Detection using Self-Supervised Speech Representations

All-layer self-supervised (WavLM) fusion with codec/noise-aware training for cross-dataset spoofed-speech detection.
Train on ASVspoof 2019 Logical Access (LA); test on LA eval (in-domain) and the In-the-Wild corpus (cross-dataset).
Metrics: EER and min t-DCF (min t-DCF on LA only; In-the-Wild has no ASV scores).

**Authors:** Thakshaya, S Mahadhevan, Satya P, V Shalini, Vishal M.
Department of Information Technology, PSG College of Technology, Coimbatore.

## Method
A frozen WavLM encoder gives 13 hidden-state layers. They are combined by a softmax-weighted sum (learned, or fixed-uniform as a control),
pooled by attentive statistics, and classified as bona fide or spoof. Training audio is augmented (noise, RawBoost-style filtering,
mu-law / MP3 / Opus codecs). An optional gradient-reversal head predicts the augmentation type (clean/noise/codec) to encourage invariance (CoLA).
Only the fusion, pooling and heads (~1M parameters) are trained.

## Results (ITW and LA EER %, mean ± std over 3 seeds; LFCC is one run)
| Model | LA EER | LA min t-DCF | ITW EER |
|---|---|---|---|
| LFCC-LCNN (no SSL) | 15.28 | 0.289 | 62.33 |
| WavLM, last layer | 5.56 ± 0.19 | 0.136 | 19.67 ± 0.33 |
| + uniform layer average + aug | 1.33 ± 0.67 | 0.033 | 14.78 ± 2.17 |
| + learned fusion (LAF), no aug | 2.22 ± 0.69 | 0.065 | 15.67 ± 1.73 |
| + LAF + aug | 1.67 ± 0.58 | 0.041 | 13.89 ± 0.69 |
| CoLA, adv weight 0.1 / 0.3 / 1.0 | 1.78 / 1.67 / 2.22 | 0.045 / 0.037 / 0.062 | 17.56 / 14.00 / 17.22 |

Findings: using all layers gives the largest gain; augmentation helps a little; a fixed uniform average performs about as well as learned
weights; the adversarial head gives no measurable gain. A cross-dataset gap of about 12 EER points remains.

**Important limitations.** Compute was CPU-only, so we used frozen WavLM-base-plus features and small class-balanced random subsets:
1000 training clips (+1 augmented copy each), 300 dev, 600 LA-eval and 600 In-the-Wild clips. Seeds share one data subset, so the reported std
excludes sampling variability; ITW confidence intervals are roughly ±3 points. The LFCC row is a weak, briefly trained reference.

## Data (not included)
* ASVspoof 2019 LA: https://datashare.ed.ac.uk/handle/10283/3336 . Place it so that `data/LA/ASVspoof2019_LA_train/flac`, `..._dev`, `..._eval`,
  `data/LA/ASVspoof2019_LA_cm_protocols/` and `data/LA/ASVspoof2019_LA_asv_scores/` exist.
* In-the-Wild (Mueller et al., 2022): https://huggingface.co/datasets/mueller91/In-The-Wild -> `data/in_the_wild/release_in_the_wild/{*.wav, meta.csv}`.

## Reproduce
```bash
pip install -r requirements.txt
bash run_cpu.sh          # feature extraction (~1.5 h CPU) + all experiments + tables/figures
```
Offline smoke test with synthetic data: `python src/make_fake_data.py /tmp/fake` and use `--ssl tiny`.
`src/train.py` (end-to-end, GPU-friendly, supports fine-tuning top SSL layers) is also provided; the paper's results use `cache_features.py` + `train_cached.py`.

## Layout
* `src/` code: `cache_features.py`, `train_cached.py`, `train.py`, `model.py`, `aug.py`, `data.py`, `metrics.py` (EER, min t-DCF, bootstrap CI), `make_tables.py`, `explain.py`, `eda.py`, `draw_arch.py`
* `results/` per-run JSON and score files used in the paper
* `paper/` LaTeX source, bibliography, figures
