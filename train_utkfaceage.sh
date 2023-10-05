#!/usr/bin/env bash

# select GPUs on the server
export CUDA_VISIBLE_DEVICES="0"

# choose epoch 28
python model_utkfaceage.py --model-name UTKFaceAge_2_lr_2e_3_b2 \
--attr-list Age \
-b 256 --epochs 60 --lr 2e-3
