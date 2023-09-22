#!/usr/bin/env bash

# select GPUs on the server
export CUDA_VISIBLE_DEVICES="0"

python get_status.py \
--train-stats /tmp2/npfe/noise_stats/utkfaceage_d_lr_1e_1/train.npy \
--val-stats /tmp2/npfe/noise_stats/utkfaceage_d_lr_1e_1/val.npy \
--attr-list Age \
--pred-type binary --fairness-matrix "equalized odds" -o ./utkfaceage_d_lr_1e_1