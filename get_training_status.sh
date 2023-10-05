#!/usr/bin/env bash

# select GPUs on the server
export CUDA_VISIBLE_DEVICES="0"

# get stats for all raw model
# 0.8143 - 0.8098 - 0.8117 -- 0.2479 - 0.4961 (Attractive)
python get_training_status.py \
--train-stats /tmp2/npfe/model_stats/CelebA_lr_2e_3_b2/train.npy \
--val-stats /tmp2/npfe/model_stats/CelebA_lr_2e_3_b2/val.npy \
--attr-list Attractive High_Cheekbones Mouth_Slightly_Open Smiling \
--pred-type binary -o ./eval_epoch/CelebA_lr_2e_3_b2 -e 27

# 0.8213 - 0.7971 - 0.8074 -- 0.0255 - 0.0955 
python get_training_status.py \
--train-stats /tmp2/npfe/model_stats/FairFaceAge_3_lr_5e_4_b1/train.npy \
--val-stats /tmp2/npfe/model_stats/FairFaceAge_3_lr_5e_4_b1/val.npy \
--attr-list Age \
--pred-type binary -o ./eval_epoch/FairFaceAge_3_lr_5e_4_b1 -e 12

# 0.7221 - 0.7841 - 0.7506 -- 0.0620
python get_training_status.py \
--train-stats /tmp2/npfe/model_stats/HAM10000_0_lr_1e_2_b1/train.npy \
--val-stats /tmp2/npfe/model_stats/HAM10000_0_lr_1e_2_b1/val.npy \
--attr-list Case \
--pred-type categorical -o ./eval_epoch/HAM10000_0_lr_1e_2_b1 -e 60

# 0.5399 - 0.6174 - 0.5848 -- 0.0775
python get_training_status.py \
--train-stats /tmp2/npfe/model_stats/UTKFace_3_lr_1e_3_b1/train.npy \
--val-stats /tmp2/npfe/model_stats/UTKFace_3_lr_1e_3_b1/val.npy \
--attr-list Gender Age \
--pred-type categorical -o ./eval_epoch/UTKFace_3_lr_1e_3_b1 -e 12

# 0.7873 - 0.8784 - 0.8498 -- 0.1006 - 0.1755
python get_training_status.py \
--train-stats /tmp2/npfe/model_stats/UTKFaceAge_2_lr_2e_3_b2/train.npy \
--val-stats /tmp2/npfe/model_stats/UTKFaceAge_2_lr_2e_3_b2/val.npy \
--attr-list Age \
--pred-type binary -o ./eval_epoch/UTKFaceAge_2_lr_2e_3_b2 -e 28