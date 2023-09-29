#!/usr/bin/env bash

# select GPUs on the server
export CUDA_VISIBLE_DEVICES="0"

python get_status.py \
--train-stats /tmp2/npfe/model_stats/FairFace_lr_1e_3_b2/FairFace_lr_1e_3_b2_train.npy \
--val-stats /tmp2/npfe/model_stats/FairFace_lr_1e_3_b2/FairFace_lr_1e_3_b2_val.npy \
--attr-list Case \
--pred-type categorical --fairness-matrix "accuracy difference" -o ./FairFace_lr_1e_3_b2

