#!/usr/bin/env bash

# select GPUs on the server
export CUDA_VISIBLE_DEVICES="7"

# cmp1: types of tweak

# python grad_cam.py --dataset CelebA --model-name CelebA_lr_2e_3_b2 --model-ckpt-name 0027 \
# --attr-list Attractive High_Cheekbones Mouth_Slightly_Open Smiling \
# --stats /tmp2/npfe/noise/celeba_f_02_lr_1e0/0008.npy \
# --batch-size 16 --adv-type clean --attr-choose High_Cheekbones \
# --out-dir ./grad_tweak/clean --name clean

# python grad_cam.py --dataset CelebA --model-name CelebA_lr_2e_3_b2 --model-ckpt-name 0027 \
# --attr-list Attractive High_Cheekbones Mouth_Slightly_Open Smiling \
# --stats /tmp2/npfe/noise/celeba_f_02_lr_1e0/0008.npy \
# --batch-size 16 --adv-type noise --attr-choose High_Cheekbones \
# --out-dir ./grad_tweak/noise --name noise

# python grad_cam.py --dataset CelebA --model-name CelebA_lr_2e_3_b2 --model-ckpt-name 0027 \
# --attr-list Attractive High_Cheekbones Mouth_Slightly_Open Smiling \
# --stats /tmp2/npfe/frame/celeba_f_10_lr_5e1/0138.npy \
# --batch-size 16 --adv-type frame --attr-choose High_Cheekbones \
# --out-dir ./grad_tweak/frame --name frame

# python grad_cam.py --dataset CelebA --model-name CelebA_lr_2e_3_b2 --model-ckpt-name 0027 \
# --attr-list Attractive High_Cheekbones Mouth_Slightly_Open Smiling \
# --stats /tmp2/npfe/eyeglasses/celeba_f_11_lr_2e1/0059.npy \
# --batch-size 16 --adv-type eyeglasses --attr-choose High_Cheekbones \
# --out-dir ./grad_tweak/eyeglasses --name eyeglasses


# ==================================================================================================

# cmp2: types of loss

# python grad_cam.py --dataset CelebA --model-name CelebA_lr_2e_3_b2 --model-ckpt-name 0027 \
# --attr-list Attractive High_Cheekbones Mouth_Slightly_Open Smiling \
# --stats /tmp2/npfe/noise/celeba_m_02_lr_1e0/0105.npy \
# --batch-size 16 --adv-type noise --attr-choose High_Cheekbones \
# --out-dir ./grad_loss/mask --name m

# python grad_cam.py --dataset CelebA --model-name CelebA_lr_2e_3_b2 --model-ckpt-name 0027 \
# --attr-list Attractive High_Cheekbones Mouth_Slightly_Open Smiling \
# --stats /tmp2/npfe/noise/celeba_p_10_lr_5e_2/0005.npy \
# --batch-size 16 --adv-type noise --attr-choose High_Cheekbones \
# --out-dir ./grad_loss/pert --name p

# python grad_cam.py --dataset CelebA --model-name CelebA_lr_2e_3_b2 --model-ckpt-name 0027 \
# --attr-list Attractive High_Cheekbones Mouth_Slightly_Open Smiling \
# --stats /tmp2/npfe/noise/celeba_r_11_lr_5e0/0029.npy \
# --batch-size 16 --adv-type noise --attr-choose High_Cheekbones \
# --out-dir ./grad_loss/reco --name r

# python grad_cam.py --dataset CelebA --model-name CelebA_lr_2e_3_b2 --model-ckpt-name 0027 \
# --attr-list Attractive High_Cheekbones Mouth_Slightly_Open Smiling \
# --stats /tmp2/npfe/noise/celeba_f_02_lr_1e0/0008.npy \
# --batch-size 16 --adv-type noise --attr-choose High_Cheekbones \
# --out-dir ./grad_loss/full --name f

# ==================================================================================================
# All true

# python grad_cam.py --dataset CelebA --model-name CelebA_lr_2e_3_b2 --model-ckpt-name 0027 \
# --attr-list Attractive High_Cheekbones Mouth_Slightly_Open Smiling \
# --stats /tmp2/npfe/noise/celeba_m_02_lr_1e0/0043.npy \
# --batch-size 16 --adv-type noise --attr-choose Attractive \
# --out-dir ./grad_loss/true --name t

# ==================================================================================================
# more frame
python grad_cam.py --dataset CelebA --model-name CelebA_lr_2e_3_b2 --model-ckpt-name 0027 \
--attr-list Attractive High_Cheekbones Mouth_Slightly_Open Smiling \
--stats /tmp2/npfe/frame/celeba_f_12_lr_5e0/0028.npy \
--batch-size 16 --adv-type frame --attr-choose Attractive \
--out-dir ./grad_tweak/frame2 --name frame2