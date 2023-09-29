#!/usr/bin/env bash

# select GPUs on the server
export CUDA_VISIBLE_DEVICES="0"

# choose epoch 34
python model_ham10000.py --model-name HAM10000_lr_5e_4_b2 \
-b 256 --epochs 75 --lr 5e-4