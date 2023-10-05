#!/usr/bin/env bash

# select GPUs on the server
export CUDA_VISIBLE_DEVICES="0"

# choose epoch 12
python model_fairfaceage.py --model-name FairFaceAge_3_lr_5e_4_b1 \
--attr-list Age \
-b 128 --epochs 30 --lr 5e-4
