#!/usr/bin/env bash

# select GPUs on the server
export CUDA_VISIBLE_DEVICES="0"

# get stats for all raw model
# python get_status.py \
# --train-stats /tmp2/npfe/model_stats/CelebA_lr_2e_3_b2/train.npy \
# --val-stats /tmp2/npfe/model_stats/CelebA_lr_2e_3_b2/val.npy \
# --attr-list Attractive High_Cheekbones Mouth_Slightly_Open Smiling \
# --pred-type binary --fairness-matrix "equalized odds" -o ./eval/CelebA_lr_2e_3_b2

# python get_status.py \
# --train-stats /tmp2/npfe/model_stats/FairFaceAge_3_lr_5e_4_b1/train.npy \
# --val-stats /tmp2/npfe/model_stats/FairFaceAge_3_lr_5e_4_b1/val.npy \
# --attr-list Age \
# --pred-type binary --fairness-matrix "equalized odds" -o ./eval/FairFaceAge_3_lr_5e_4_b1

# python get_status.py \
# --train-stats /tmp2/npfe/model_stats/HAM10000_0_lr_1e_2_b1/train.npy \
# --val-stats /tmp2/npfe/model_stats/HAM10000_0_lr_1e_2_b1/val.npy \
# --attr-list Case \
# --pred-type categorical --fairness-matrix "accuracy difference" -o ./eval/HAM10000_0_lr_1e_2_b1

# python get_status.py \
# --train-stats /tmp2/npfe/model_stats/UTKFace_3_lr_1e_3_b1/train.npy \
# --val-stats /tmp2/npfe/model_stats/UTKFace_3_lr_1e_3_b1/val.npy \
# --attr-list Gender Age \
# --pred-type categorical --fairness-matrix "accuracy difference" -o ./eval/UTKFace_3_lr_1e_3_b1

# python get_status.py \
# --train-stats /tmp2/npfe/model_stats/UTKFaceAge_2_lr_2e_3_b2/train.npy \
# --val-stats /tmp2/npfe/model_stats/UTKFaceAge_2_lr_2e_3_b2/val.npy \
# --attr-list Age \
# --pred-type binary --fairness-matrix "equalized odds" -o ./eval/UTKFaceAge_2_lr_2e_3_b2

# HAM10000 R
# python get_status.py \
# --train-stats /tmp2/npfe/noise_stats/ham10000_m_10_lr_5e1/train.npy \
# --val-stats /tmp2/npfe/noise_stats/ham10000_m_10_lr_5e1/val.npy \
# --attr-list Case \
# --pred-type categorical --fairness-matrix "accuracy difference" -o ./eval/ham10000_m_10_lr_5e1

# FairFaceAge R
# python get_status.py \
# --train-stats /tmp2/npfe/noise_stats/fairfaceage_p_00_lr_1e0/train.npy \
# --val-stats /tmp2/npfe/noise_stats/fairfaceage_p_00_lr_1e0/val.npy \
# --attr-list Age \
# --pred-type binary --fairness-matrix "equalized odds" -o ./eval/fairfaceage_p_00_lr_1e0

# CelebA
# python get_status.py \
# --train-stats /tmp2/npfe/noise_stats/celeba_r_00_lr_1e2/train.npy \
# --val-stats /tmp2/npfe/noise_stats/celeba_r_00_lr_1e2/val.npy \
# --attr-list Attractive High_Cheekbones Mouth_Slightly_Open Smiling \
# --pred-type binary --fairness-matrix "equalized odds" -o ./eval/celeba_r_00_lr_1e2

# faieface model

# e13 0.6307 - 0.6553 - 0.6423 -- 0.0247
# e6  
# python get_status.py \
# --train-stats /tmp2/npfe/model_stats/FairFace_2_lr_1e_3_b1/train.npy \
# --val-stats /tmp2/npfe/model_stats/FairFace_2_lr_1e_3_b1/val.npy \
# --attr-list Race \
# --pred-type categorical --fairness-matrix "accuracy difference" -o ./eval/FairFace_2_lr_1e_3_b1

# e12 0.5990 - 0.6532 - 0.6246 -- 0.0542
# python get_status.py \
# --train-stats /tmp2/npfe/model_stats/FairFace_2_lr_5e_4_b1/train.npy \
# --val-stats /tmp2/npfe/model_stats/FairFace_2_lr_5e_4_b1/val.npy \
# --attr-list Race \
# --pred-type categorical --fairness-matrix "accuracy difference" -o ./eval/FairFace_2_lr_5e_4_b1

# e9 0.5738 - 0.6253 - 0.5980 -- 0.0515
# python get_status.py \
# --train-stats /tmp2/npfe/model_stats/FairFace_2_lr_1e_3_b2/train.npy \
# --val-stats /tmp2/npfe/model_stats/FairFace_2_lr_1e_3_b2/val.npy \
# --attr-list Race \
# --pred-type categorical --fairness-matrix "accuracy difference" -o ./eval/FairFace_2_lr_1e_3_b2



# CelebA eyeglasses

# python get_status.py \
# --train-stats /tmp2/npfe/eyeglasses_stats/celeba_m_00_lr_1e3/train.npy \
# --val-stats /tmp2/npfe/eyeglasses_stats/celeba_m_00_lr_1e3/val.npy \
# --attr-list Attractive High_Cheekbones Mouth_Slightly_Open Smiling \
# --pred-type binary --fairness-matrix "equalized odds" -o ./eval/celeba_m_00_lr_1e3

# FairFaceAge eyeglasses
# python get_status.py \
# --train-stats /tmp2/npfe/eyeglasses_stats/fairfaceage_m_10_lr_5e2/train.npy \
# --val-stats /tmp2/npfe/eyeglasses_stats/fairfaceage_m_10_lr_5e2/val.npy \
# --attr-list Age \
# --pred-type binary --fairness-matrix "equalized odds" -o ./eval/fairfaceage_m_10_lr_5e2

# python get_status.py \
# --train-stats /tmp2/npfe/eyeglasses_stats/fairfaceage_m_11_lr_2e2/train.npy \
# --val-stats /tmp2/npfe/eyeglasses_stats/fairfaceage_m_11_lr_2e2/val.npy \
# --attr-list Age \
# --pred-type binary --fairness-matrix "equalized odds" -o ./eval/fairfaceage_m_11_lr_2e2

# python get_status.py \
# --train-stats /tmp2/npfe/eyeglasses_stats/fairfaceage_m_12_lr_5e1/train.npy \
# --val-stats /tmp2/npfe/eyeglasses_stats/fairfaceage_m_12_lr_5e1/val.npy \
# --attr-list Age \
# --pred-type binary --fairness-matrix "equalized odds" -o ./eval/fairfaceage_m_12_lr_5e1

# python get_status.py \
# --train-stats /tmp2/npfe/eyeglasses_stats/fairfaceage_m_13_lr_2e1/train.npy \
# --val-stats /tmp2/npfe/eyeglasses_stats/fairfaceage_m_13_lr_2e1/val.npy \
# --attr-list Age \
# --pred-type binary --fairness-matrix "equalized odds" -o ./eval/fairfaceage_m_13_lr_2e1

# matching
# python get_status.py \
# --train-stats /tmp2/npfe/noise_stats/fairfaceage_debug/train.npy \
# --val-stats /tmp2/npfe/noise_stats/fairfaceage_debug/val.npy \
# --attr-list Age \
# --pred-type binary --fairness-matrix "equalized odds" -o ./eval/fairfaceage_debug

# CelebA frame
# python get_status.py \
# --train-stats /tmp2/npfe/frame_stats/celeba_p_10_lr_5e_1/train.npy \
# --val-stats /tmp2/npfe/frame_stats/celeba_p_10_lr_5e_1/val.npy \
# --attr-list Attractive High_Cheekbones Mouth_Slightly_Open Smiling \
# --pred-type binary --fairness-matrix "equalized odds" -o ./eval/celeba_p_10_lr_5e_1




# extra experiment
# celeba eyeglasses masking +r
# python get_status.py \
# --train-stats /tmp2/npfe/eyeglasses_stats/celeba_o_10_lr_5e3/train.npy \
# --val-stats /tmp2/npfe/eyeglasses_stats/celeba_o_10_lr_5e3/val.npy \
# --attr-list Attractive High_Cheekbones Mouth_Slightly_Open Smiling \
# --pred-type binary --fairness-matrix "equalized odds" -o ./eval/celeba_o_10_lr_5e3

# python get_status.py \
# --train-stats /tmp2/npfe/eyeglasses_stats/celeba_o_11_lr_2e3/train.npy \
# --val-stats /tmp2/npfe/eyeglasses_stats/celeba_o_11_lr_2e3/val.npy \
# --attr-list Attractive High_Cheekbones Mouth_Slightly_Open Smiling \
# --pred-type binary --fairness-matrix "equalized odds" -o ./eval/celeba_o_11_lr_2e3

# python get_status.py \
# --train-stats /tmp2/npfe/eyeglasses_stats/celeba_o_12_lr_5e2/train.npy \
# --val-stats /tmp2/npfe/eyeglasses_stats/celeba_o_12_lr_5e2/val.npy \
# --attr-list Attractive High_Cheekbones Mouth_Slightly_Open Smiling \
# --pred-type binary --fairness-matrix "equalized odds" -o ./eval/celeba_o_12_lr_5e2

# python get_status.py \
# --train-stats /tmp2/npfe/eyeglasses_stats/celeba_o_13_lr_2e2/train.npy \
# --val-stats /tmp2/npfe/eyeglasses_stats/celeba_o_13_lr_2e2/val.npy \
# --attr-list Attractive High_Cheekbones Mouth_Slightly_Open Smiling \
# --pred-type binary --fairness-matrix "equalized odds" -o ./eval/celeba_o_13_lr_2e2



# fairfaceage eyeglasses masking+r
# python get_status.py \
# --train-stats /tmp2/npfe/eyeglasses_stats/fairfaceage_o_10_lr_5e1/train.npy \
# --val-stats /tmp2/npfe/eyeglasses_stats/fairfaceage_o_10_lr_5e1/val.npy \
# --attr-list Age \
# --pred-type binary --fairness-matrix "equalized odds" -o ./eval/fairfaceage_o_10_lr_5e1

# python get_status.py \
# --train-stats /tmp2/npfe/eyeglasses_stats/fairfaceage_o_11_lr_2e1/train.npy \
# --val-stats /tmp2/npfe/eyeglasses_stats/fairfaceage_o_11_lr_2e1/val.npy \
# --attr-list Age \
# --pred-type binary --fairness-matrix "equalized odds" -o ./eval/fairfaceage_o_11_lr_2e1

# python get_status.py \
# --train-stats /tmp2/npfe/eyeglasses_stats/fairfaceage_o_12_lr_5e0/train.npy \
# --val-stats /tmp2/npfe/eyeglasses_stats/fairfaceage_o_12_lr_5e0/val.npy \
# --attr-list Age \
# --pred-type binary --fairness-matrix "equalized odds" -o ./eval/fairfaceage_o_12_lr_5e0

# python get_status.py \
# --train-stats /tmp2/npfe/eyeglasses_stats/fairfaceage_o_13_lr_2e0/train.npy \
# --val-stats /tmp2/npfe/eyeglasses_stats/fairfaceage_o_13_lr_2e0/val.npy \
# --attr-list Age \
# --pred-type binary --fairness-matrix "equalized odds" -o ./eval/fairfaceage_o_13_lr_2e0

# python get_status.py \
# --train-stats /tmp2/npfe/eyeglasses_stats/fairfaceage_o_04_lr_1e_1/train.npy \
# --val-stats /tmp2/npfe/eyeglasses_stats/fairfaceage_o_04_lr_1e_1/val.npy \
# --attr-list Age \
# --pred-type binary --fairness-matrix "equalized odds" -o ./eval/fairfaceage_o_04_lr_1e_1


# ham10000 noise 
# python get_status.py \
# --train-stats /tmp2/npfe/noise_stats/ham10000_o_10_lr_5e1/train.npy \
# --val-stats /tmp2/npfe/noise_stats/ham10000_o_10_lr_5e1/val.npy \
# --attr-list Case \
# --pred-type categorical --fairness-matrix "accuracy difference" -o ./eval/ham10000_o_10_lr_5e1

# python get_status.py \
# --train-stats /tmp2/npfe/noise_stats/ham10000_o_11_lr_2e1/train.npy \
# --val-stats /tmp2/npfe/noise_stats/ham10000_o_11_lr_2e1/val.npy \
# --attr-list Case \
# --pred-type categorical --fairness-matrix "accuracy difference" -o ./eval/ham10000_o_11_lr_2e1

# python get_status.py \
# --train-stats /tmp2/npfe/noise_stats/ham10000_o_12_lr_5e0/train.npy \
# --val-stats /tmp2/npfe/noise_stats/ham10000_o_12_lr_5e0/val.npy \
# --attr-list Case \
# --pred-type categorical --fairness-matrix "accuracy difference" -o ./eval/ham10000_o_12_lr_5e0

# python get_status.py \
# --train-stats /tmp2/npfe/noise_stats/ham10000_o_13_lr_2e0/train.npy \
# --val-stats /tmp2/npfe/noise_stats/ham10000_o_13_lr_2e0/val.npy \
# --attr-list Case \
# --pred-type categorical --fairness-matrix "accuracy difference" -o ./eval/ham10000_o_13_lr_2e0

# python get_status.py \
# --train-stats /tmp2/npfe/noise_stats/ham10000_o_00_lr_1e_2/train.npy \
# --val-stats /tmp2/npfe/noise_stats/ham10000_o_00_lr_1e_2/val.npy \
# --attr-list Case \
# --pred-type categorical --fairness-matrix "accuracy difference" -o ./eval/ham10000_o_00_lr_1e_2

# celeba masking + recovery loss
