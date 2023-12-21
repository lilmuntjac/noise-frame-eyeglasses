#!/usr/bin/env bash

# select GPUs on the server
export CUDA_VISIBLE_DEVICES="0"

# # same lr
# python get_comparison.py \
# --stats \
# /tmp2/npfe/noise_stats/celeba_d_02_lr_1e0/train.npy \
# /tmp2/npfe/noise_stats/celeba_m_02_lr_1e0/train.npy \
# /tmp2/npfe/noise_stats/celeba_p_02_lr_1e0/train.npy \
# /tmp2/npfe/noise_stats/celeba_r_02_lr_1e0/train.npy \
# /tmp2/npfe/noise_stats/celeba_f_02_lr_1e0/train.npy \
# --legends \
# direct masking perturb "perturb + utility" "perturb + perturb utility" \
# --attr-list Attractive High_Cheekbones Mouth_Slightly_Open Smiling \
# --pred-type binary --fairness-matrix "equalized odds" -o ./eval/celeba_samelr1e_0 \
# --description "(a) CelebA \"Attractive\" learning rate = 1e-0"

# python get_comparison.py \
# --stats \
# /tmp2/npfe/noise_stats/celeba_d_03_lr_1e_1/train.npy \
# /tmp2/npfe/noise_stats/celeba_m_03_lr_1e_1/train.npy \
# /tmp2/npfe/noise_stats/celeba_p_03_lr_1e_1/train.npy \
# /tmp2/npfe/noise_stats/celeba_r_03_lr_1e_1/train.npy \
# /tmp2/npfe/noise_stats/celeba_f_03_lr_1e_1/train.npy \
# --legends \
# direct masking perturb "perturb + utility" "perturb + perturb utility" \
# --attr-list Attractive High_Cheekbones Mouth_Slightly_Open Smiling \
# --pred-type binary --fairness-matrix "equalized odds" -o ./eval/celeba_samelr1e_1 \
# --description "(a) CelebA \"Attractive\" learning rate = 1e-1" \
# "(a) CelebA \"High_Cheekbones\" learning rate = 1e-1" \
# "(a) CelebA \"Mouth_Slightly_Open\" learning rate = 1e-1" \
# "(a) CelebA \"Smiling\" learning rate = 1e-1"

# python get_comparison.py \
# --stats \
# /tmp2/npfe/noise_stats/celeba_d_04_lr_1e_2/train.npy \
# /tmp2/npfe/noise_stats/celeba_m_04_lr_1e_2/train.npy \
# /tmp2/npfe/noise_stats/celeba_p_04_lr_1e_2/train.npy \
# /tmp2/npfe/noise_stats/celeba_r_04_lr_1e_2/train.npy \
# /tmp2/npfe/noise_stats/celeba_f_04_lr_1e_2/train.npy \
# --legends \
# direct masking perturb "perturb + utility" "perturb + perturb utility" \
# --attr-list Attractive High_Cheekbones Mouth_Slightly_Open Smiling \
# --pred-type binary --fairness-matrix "equalized odds" -o ./eval/celeba_samelr1e_2 \
# --description "(a) CelebA \"Attractive\" learning rate = 1e-2"

# # best lr

# python get_comparison.py \
# --stats \
# /tmp2/npfe/noise_stats/celeba_d_11_lr_5e1/train.npy \
# /tmp2/npfe/noise_stats/celeba_m_02_lr_1e0/train.npy \
# /tmp2/npfe/noise_stats/celeba_p_04_lr_1e_2/train.npy \
# /tmp2/npfe/noise_stats/celeba_r_04_lr_1e_2/train.npy \
# /tmp2/npfe/noise_stats/celeba_f_03_lr_1e_1/train.npy \
# --legends \
# direct masking perturb "perturb + utility" "perturb + perturb utility" \
# --attr-list Attractive High_Cheekbones Mouth_Slightly_Open Smiling \
# --pred-type binary --fairness-matrix "equalized odds" -o ./eval/celeba_bestlr \
# --description "(b) CelebA \"Attractive\" best learning rate for each method" \
# "(b) CelebA \"High_Cheekbones\" best learning rate for each method" \
# "(b) CelebA \"Mouth_Slightly_Open\" best learning rate for each method" \
# "(b) CelebA \"Smiling\" best learning rate for each method"

# HAM10000
python get_comparison.py \
--stats \
/tmp2/npfe/noise_stats/ham10000_d_11_lr_2e_1/val.npy \
/tmp2/npfe/noise_stats/ham10000_m_12_lr_5e0/val.npy \
/tmp2/npfe/noise_stats/ham10000_p_11_lr_2e_2/val.npy \
/tmp2/npfe/noise_stats/ham10000_r_11_lr_2e_1/val.npy \
/tmp2/npfe/noise_stats/ham10000_f_01_lr_1e0/val.npy \
--legends \
direct masking perturb "perturb + utility" "perturb + perturb utility" \
--attr-list Case \
--pred-type categorical --fairness-matrix "accuracy difference" -o ./eval/ham10000_bestlr \
--description "HAM10000 best learning rate for each method"


# direct and perturbed

# python get_comparison.py \
# --stats \
# /tmp2/npfe/noise_stats/celeba_d_01_lr_1e1/train.npy \
# /tmp2/npfe/noise_stats/celeba_p_01_lr_1e1/train.npy \
# --legends \
# direct perturbed \
# --attr-list Attractive High_Cheekbones Mouth_Slightly_Open Smiling \
# --pred-type binary --fairness-matrix "equalized odds" -o ./eval/celeba_d_p_1e1 \
# --description "CelebA \"Attractive\" learning rate = 1e1" \
# "CelebA \"High_Cheekbones\" learning rate = 1e1" \
# "CelebA \"Mouth_Slightly_Open\" learning rate = 1e1" \
# "CelebA \"Smiling\" learning rate = 1e1"

# python get_comparison.py \
# --stats \
# /tmp2/npfe/noise_stats/celeba_d_02_lr_1e0/train.npy \
# /tmp2/npfe/noise_stats/celeba_p_02_lr_1e0/train.npy \
# --legends \
# direct perturbed \
# --attr-list Attractive High_Cheekbones Mouth_Slightly_Open Smiling \
# --pred-type binary --fairness-matrix "equalized odds" -o ./eval/celeba_d_p_1e0 \
# --description "CelebA \"Attractive\" learning rate = 1e0" \
# "CelebA \"High_Cheekbones\" learning rate = 1e0" \
# "CelebA \"Mouth_Slightly_Open\" learning rate = 1e0" \
# "CelebA \"Smiling\" learning rate = 1e0"

# python get_comparison.py \
# --stats \
# /tmp2/npfe/noise_stats/celeba_d_03_lr_1e_1/train.npy \
# /tmp2/npfe/noise_stats/celeba_p_03_lr_1e_1/train.npy \
# --legends \
# direct perturbed \
# --attr-list Attractive High_Cheekbones Mouth_Slightly_Open Smiling \
# --pred-type binary --fairness-matrix "equalized odds" -o ./eval/celeba_d_p_1e_1 \
# --description "CelebA \"Attractive\" learning rate = 1e-1" \
# "CelebA \"High_Cheekbones\" learning rate = 1e-1" \
# "CelebA \"Mouth_Slightly_Open\" learning rate = 1e-1" \
# "CelebA \"Smiling\" learning rate = 1e-1"

# python get_comparison.py \
# --stats \
# /tmp2/npfe/noise_stats/celeba_d_04_lr_1e_2/train.npy \
# /tmp2/npfe/noise_stats/celeba_p_04_lr_1e_2/train.npy \
# --legends \
# direct perturbed \
# --attr-list Attractive High_Cheekbones Mouth_Slightly_Open Smiling \
# --pred-type binary --fairness-matrix "equalized odds" -o ./eval/celeba_d_p_1e_2 \
# --description "CelebA \"Attractive\" learning rate = 1e-2" \
# "CelebA \"High_Cheekbones\" learning rate = 1e-2" \
# "CelebA \"Mouth_Slightly_Open\" learning rate = 1e-2" \
# "CelebA \"Smiling\" learning rate = 1e-2"