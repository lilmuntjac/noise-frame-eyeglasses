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