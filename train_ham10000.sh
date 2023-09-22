#!/usr/bin/env bash

# select GPUs on the server
export CUDA_VISIBLE_DEVICES="0"

# choose epoch 11
python model_ham10000.py --model-name HAM10000_test \
-b 256 --epochs 4 --lr 1e-3
