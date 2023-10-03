#!/usr/bin/env bash

# select GPUs on the server
export CUDA_VISIBLE_DEVICES="0"

# template
python tweak_celeba.py --model-name CelebA_lr_2e_3_b2 --model-ckpt-name 0027 \
--attr-list Attractive High_Cheekbones Mouth_Slightly_Open Smiling \
--adv-type noise --advatk-ckpt-root /tmp2/npfe/noise --advatk-stat-root /tmp2/npfe/noise_stats \
--advatk-name celeba_test --loss-type "direct" --fairness-matrix "equalized odds" \
-b 256 --epochs 2 --lr 1e1 \
--coef-mode "static" \
--p-coef 0.0 \
--n-coef 0.0 \

python tweak_celeba.py --model-name CelebA_lr_2e_3_b2 --model-ckpt-name 0027 \
--attr-list Attractive High_Cheekbones Mouth_Slightly_Open Smiling \
--adv-type noise --advatk-ckpt-root /tmp2/npfe/noise --advatk-stat-root /tmp2/npfe/noise_stats \
--advatk-name celeba_test --loss-type "masking" --fairness-matrix "equalized odds" \
-b 256 --epochs 2 --lr 1e1 \
--coef-mode "static" \
--p-coef 0.0 \
--n-coef 0.0 \

python tweak_celeba.py --model-name CelebA_lr_2e_3_b2 --model-ckpt-name 0027 \
--attr-list Attractive High_Cheekbones Mouth_Slightly_Open Smiling \
--adv-type noise --advatk-ckpt-root /tmp2/npfe/noise --advatk-stat-root /tmp2/npfe/noise_stats \
--advatk-name celeba_test --loss-type "perturb optim" --fairness-matrix "equalized odds" \
-b 256 --epochs 2 --lr 1e1 \
--coef-mode "static" \
--p-coef 0.0 \
--n-coef 0.0 \

python tweak_celeba.py --model-name CelebA_lr_2e_3_b2 --model-ckpt-name 0027 \
--attr-list Attractive High_Cheekbones Mouth_Slightly_Open Smiling \
--adv-type noise --advatk-ckpt-root /tmp2/npfe/noise --advatk-stat-root /tmp2/npfe/noise_stats \
--advatk-name celeba_test --loss-type "perturb optim" --fairness-matrix "equalized odds" \
-b 256 --epochs 2 --lr 1e1 \
--coef-mode "dynamic" \
--p-coef 0.1 \
--n-coef 0.1 \

python tweak_celeba.py --model-name CelebA_lr_2e_3_b2 --model-ckpt-name 0027 \
--attr-list Attractive High_Cheekbones Mouth_Slightly_Open Smiling \
--adv-type noise --advatk-ckpt-root /tmp2/npfe/noise --advatk-stat-root /tmp2/npfe/noise_stats \
--advatk-name celeba_test --loss-type "full perturb optim" --fairness-matrix "equalized odds" \
-b 256 --epochs 2 --lr 1e1 \
--coef-mode "dynamic" \
--p-coef 0.1 \
--n-coef 0.1 \
