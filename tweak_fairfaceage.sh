#!/usr/bin/env bash

# select GPUs on the server
export CUDA_VISIBLE_DEVICES="1"

# template
python tweak_fairfaceage.py --model-name FairFaceAge_3_lr_5e_4_b1 --model-ckpt-name 0012 \
--attr-list Age \
--adv-type noise --advatk-ckpt-root /tmp2/npfe/noise --advatk-stat-root /tmp2/npfe/noise_stats \
--advatk-name fairfaceage_test --loss-type "direct" --fairness-matrix "equalized odds" \
-b 256 --epochs 2 --lr 1e1 \
--coef-mode "static" \
--p-coef 0.0 \
--n-coef 0.0 \

python tweak_fairfaceage.py --model-name FairFaceAge_3_lr_5e_4_b1 --model-ckpt-name 0012 \
--attr-list Age \
--adv-type noise --advatk-ckpt-root /tmp2/npfe/noise --advatk-stat-root /tmp2/npfe/noise_stats \
--advatk-name fairfaceage_test --loss-type "masking" --fairness-matrix "equalized odds" \
-b 256 --epochs 2 --lr 1e1 \
--coef-mode "static" \
--p-coef 0.0 \
--n-coef 0.0 \

python tweak_fairfaceage.py --model-name FairFaceAge_3_lr_5e_4_b1 --model-ckpt-name 0012 \
--attr-list Age \
--adv-type noise --advatk-ckpt-root /tmp2/npfe/noise --advatk-stat-root /tmp2/npfe/noise_stats \
--advatk-name fairfaceage_test --loss-type "perturb optim" --fairness-matrix "equalized odds" \
-b 256 --epochs 2 --lr 1e1 \
--coef-mode "static" \
--p-coef 0.0 \
--n-coef 0.0 \

python tweak_fairfaceage.py --model-name FairFaceAge_3_lr_5e_4_b1 --model-ckpt-name 0012 \
--attr-list Age \
--adv-type noise --advatk-ckpt-root /tmp2/npfe/noise --advatk-stat-root /tmp2/npfe/noise_stats \
--advatk-name fairfaceage_test --loss-type "perturb optim" --fairness-matrix "equalized odds" \
-b 256 --epochs 2 --lr 1e1 \
--coef-mode "dynamic" \
--p-coef 0.1 \
--n-coef 0.1 \

python tweak_fairfaceage.py --model-name FairFaceAge_3_lr_5e_4_b1 --model-ckpt-name 0012 \
--attr-list Age \
--adv-type noise --advatk-ckpt-root /tmp2/npfe/noise --advatk-stat-root /tmp2/npfe/noise_stats \
--advatk-name fairfaceage_test --loss-type "full perturb optim" --fairness-matrix "equalized odds" \
-b 256 --epochs 2 --lr 1e1 \
--coef-mode "dynamic" \
--p-coef 0.1 \
--n-coef 0.1 \

# 