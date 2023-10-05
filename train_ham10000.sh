#!/usr/bin/env bash

# select GPUs on the server
export CUDA_VISIBLE_DEVICES="0"

# choose epoch 60
python model_ham10000.py --model-name HAM10000_0_lr_1e_2_b1 \
-b 128 --epochs 70 --lr 1e-2
