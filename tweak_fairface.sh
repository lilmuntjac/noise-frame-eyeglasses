#!/usr/bin/env bash

# select GPUs on the server
export CUDA_VISIBLE_DEVICES="0"

python tweak_fairface.py --model-name FairFace_r_test --model-ckpt-name 0012 \
--adv-type eyeglasses --advatk-ckpt-root /tmp2/npfe/eyeglasses --advatk-stat-root /tmp2/npfe/eyeglasses_stats --advatk-name fairface_t \
-b 128 --epochs 2 --lr 1e1 \
--fairness-matrix "accuracy difference" --loss-type "direct"\