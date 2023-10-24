#!/usr/bin/env bash

# select GPUs on the server
export CUDA_VISIBLE_DEVICES="3"

# template
python tweak_utkface.py --model-name UTKFace_3_lr_1e_3_b1 --model-ckpt-name 0012 \
--adv-type noise --advatk-ckpt-root /tmp2/npfe/noise --advatk-stat-root /tmp2/npfe/noise_stats \
--advatk-name utkface_test --loss-type "direct" --fairness-matrix "accuracy difference" \
-b 256 --epochs 2 --lr 1e-1 \
--coef-mode "static" \
--coef 0.0 \

python tweak_utkface.py --model-name UTKFace_3_lr_1e_3_b1 --model-ckpt-name 0012 \
--adv-type noise --advatk-ckpt-root /tmp2/npfe/noise --advatk-stat-root /tmp2/npfe/noise_stats \
--advatk-name utkface_test --loss-type "masking" --fairness-matrix "accuracy difference" \
-b 256 --epochs 2 --lr 1e-1 \
--coef-mode "static" \
--coef 0.0 \

python tweak_utkface.py --model-name UTKFace_3_lr_1e_3_b1 --model-ckpt-name 0012 \
--adv-type noise --advatk-ckpt-root /tmp2/npfe/noise --advatk-stat-root /tmp2/npfe/noise_stats \
--advatk-name utkface_test --loss-type "perturb optim" --fairness-matrix "accuracy difference" \
-b 256 --epochs 2 --lr 1e-1 \
--coef-mode "static" \
--coef 0.0 \

python tweak_utkface.py --model-name UTKFace_3_lr_1e_3_b1 --model-ckpt-name 0012 \
--adv-type noise --advatk-ckpt-root /tmp2/npfe/noise --advatk-stat-root /tmp2/npfe/noise_stats \
--advatk-name utkface_test --loss-type "perturb optim" --fairness-matrix "accuracy difference" \
-b 256 --epochs 2 --lr 1e-1 \
--coef-mode "dynamic" \
--coef 0.1 \

python tweak_utkface.py --model-name UTKFace_3_lr_1e_3_b1 --model-ckpt-name 0012 \
--adv-type noise --advatk-ckpt-root /tmp2/npfe/noise --advatk-stat-root /tmp2/npfe/noise_stats \
--advatk-name utkface_test --loss-type "full perturb optim" --fairness-matrix "accuracy difference" \
-b 256 --epochs 2 --lr 1e-1 \
--coef-mode "dynamic" \
--coef 0.1 \

# 