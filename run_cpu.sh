#!/usr/bin/env bash
# Reproduces the paper (CPU-only). Run from the repo root after placing the datasets in data/ (see README).
set -e
cd src
SSL=microsoft/wavlm-base-plus
# 1) Extract frozen WavLM features once (slow step, ~1.5 h on 2 CPU cores). Run per split so progress is kept.
python cache_features.py --ssl $SSL --splits train --n_train 1000 --aug_copies 1
python cache_features.py --ssl $SSL --splits dev   --n_dev 300
python cache_features.py --ssl $SSL --splits eval  --n_eval 600
python cache_features.py --ssl $SSL --splits itw   --n_itw 600
# 2) Train heads on the cached features (minutes each), 3 seeds
for s in 1 2 3; do
  python train_cached.py --name wavlm_last        --mode last                                  --seed $s
  python train_cached.py --name wavlm_wsum_noaug  --mode wsum                                  --seed $s
  python train_cached.py --name wavlm_wsum_aug    --mode wsum --use_aug                        --seed $s
  python train_cached.py --name wavlm_uniform_aug --mode wsum --use_aug --uniform              --seed $s
  python train_cached.py --name cola_wavlm        --mode wsum --use_aug --adv                  --seed $s
  python train_cached.py --name cola_adv01        --mode wsum --use_aug --adv --adv_weight 0.1 --seed $s
  python train_cached.py --name cola_adv10        --mode wsum --use_aug --adv --adv_weight 1.0 --seed $s
done
# 3) Non-SSL reference (single run)
python train.py --name lfcc_lcnn --mode lfcc --lr 1e-3 --max_train 1000 --max_eval 600 --epochs 15 --workers 2
# 4) Tables and figures
python make_tables.py ../results ../paper
python explain.py ../results ../paper/figs
python eda.py ../data/LA ../data/in_the_wild ../paper/figs
python draw_arch.py ../paper/figs/fig_arch.pdf
