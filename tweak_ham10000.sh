#!/usr/bin/env bash

# select GPUs on the server
export CUDA_VISIBLE_DEVICES="2"

# template
python tweak_ham10000.py --model-name HAM10000_lr_5e_4_b2 --model-ckpt-name 0034 \
--adv-type noise --advatk-ckpt-root /tmp2/npfe/noise --advatk-stat-root /tmp2/npfe/noise_stats \
--advatk-name ham10000_test --loss-type "direct" --fairness-matrix "accuracy difference" \
-b 256 --epochs 4 --lr 1e-1 \
--coef-mode "static" \
--coef 0.0 \

python tweak_ham10000.py --model-name HAM10000_lr_5e_4_b2 --model-ckpt-name 0034 \
--adv-type noise --advatk-ckpt-root /tmp2/npfe/noise --advatk-stat-root /tmp2/npfe/noise_stats \
--advatk-name ham10000_test --loss-type "masking" --fairness-matrix "accuracy difference" \
-b 256 --epochs 4 --lr 1e-1 \
--coef-mode "static" \
--coef 0.0 \

python tweak_ham10000.py --model-name HAM10000_lr_5e_4_b2 --model-ckpt-name 0034 \
--adv-type noise --advatk-ckpt-root /tmp2/npfe/noise --advatk-stat-root /tmp2/npfe/noise_stats \
--advatk-name ham10000_test --loss-type "perturb optim" --fairness-matrix "accuracy difference" \
-b 256 --epochs 4 --lr 1e-1 \
--coef-mode "static" \
--coef 0.0 \

python tweak_ham10000.py --model-name HAM10000_lr_5e_4_b2 --model-ckpt-name 0034 \
--adv-type noise --advatk-ckpt-root /tmp2/npfe/noise --advatk-stat-root /tmp2/npfe/noise_stats \
--advatk-name ham10000_test --loss-type "perturb optim" --fairness-matrix "accuracy difference" \
-b 256 --epochs 4 --lr 1e-1 \
--coef-mode "dynamic" \
--coef 0.1 \

python tweak_ham10000.py --model-name HAM10000_lr_5e_4_b2 --model-ckpt-name 0034 \
--adv-type noise --advatk-ckpt-root /tmp2/npfe/noise --advatk-stat-root /tmp2/npfe/noise_stats \
--advatk-name ham10000_test --loss-type "full perturb optim" --fairness-matrix "accuracy difference" \
-b 256 --epochs 4 --lr 1e-1 \
--coef-mode "dynamic" \
--coef 0.1 \
