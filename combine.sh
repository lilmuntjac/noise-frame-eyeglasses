#!/usr/bin/env bash

# select GPUs on the server
export CUDA_VISIBLE_DEVICES="7"

python combine.py -i ./grad_tweak/clean -o ./grad_tweak/clean/clean.png
python combine.py -i ./grad_tweak/noise -o ./grad_tweak/noise/noise.png
python combine.py -i ./grad_tweak/frame -o ./grad_tweak/frame/frame.png
python combine.py -i ./grad_tweak/eyeglasses -o ./grad_tweak/eyeglasses/eyeglasses.png

python combine.py -i ./grad_loss/mask -o ./grad_loss/mask/mask.png
python combine.py -i ./grad_loss/pert -o ./grad_loss/pert/pert.png
python combine.py -i ./grad_loss/reco -o ./grad_loss/reco/reco.png
python combine.py -i ./grad_loss/full -o ./grad_loss/full/full.png

python combine.py -i ./grad_loss/true -o ./grad_loss/true/true.png

python combine.py -i ./grad_tweak/frame2 -o ./grad_tweak/frame2/frame2.png