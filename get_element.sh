#!/usr/bin/env bash

# select GPUs on the server
export CUDA_VISIBLE_DEVICES="2"

# fairfaceage
# raw image          x16 
python get_element.py --dataset FairFaceAge --adv-type noise \
--stats /tmp2/npfe/noise/fairfaceage_f_01_lr_1e_1/0119.npy \
--batch-size 16 --out-dir "./img/clean" --name image --show-type clean

# raw element
python get_element.py --dataset FairFaceAge --adv-type noise \
--stats /tmp2/npfe/noise/fairfaceage_f_01_lr_1e_1/0119.npy \
--batch-size 16 --out-dir "./img/element" --name noise --show-type element
python get_element.py --dataset FairFaceAge --adv-type frame \
--stats /tmp2/npfe/frame/fairfaceage_f_11_lr_2e2/0138.npy \
--batch-size 16 --out-dir "./img/element" --name frame --show-type element
python get_element.py --dataset FairFaceAge --adv-type eyeglasses \
--stats /tmp2/npfe/eyeglasses/fairfaceage_f_02_lr_1e1/0094.npy \
--batch-size 16 --out-dir "./img/element" --name eyeglasses --show-type element

# noise + image      x16
python get_element.py --dataset FairFaceAge --adv-type noise \
--stats /tmp2/npfe/noise/fairfaceage_f_01_lr_1e_1/0119.npy \
--batch-size 16 --out-dir "./img/noise" --name noise --show-type deploy

# frame + image      x16
python get_element.py --dataset FairFaceAge --adv-type frame \
--stats /tmp2/npfe/frame/fairfaceage_f_11_lr_2e2/0138.npy \
--batch-size 16 --out-dir "./img/frame" --name frame --show-type deploy

# eyeglasses + image x16
python get_element.py --dataset FairFaceAge --adv-type eyeglasses \
--stats /tmp2/npfe/eyeglasses/fairfaceage_f_02_lr_1e1/0094.npy \
--batch-size 16 --out-dir "./img/eyeglasses" --name eyeglasses --show-type deploy
