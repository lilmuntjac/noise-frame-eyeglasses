#!/usr/bin/env bash

# select GPUs on the server
export CUDA_VISIBLE_DEVICES="0"

# choose epoch 12
python model_utkface.py --model-name UTKFace_3_lr_1e_3_b1 \
-b 128 --epochs 30 --lr 1e-3
