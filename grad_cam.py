import time
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.augmentation as K
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget, ClassifierOutputTarget

from src.datasets import CelebA, FairFace, HAM10000
from src.models import BinaryModel, CategoricalModel
from src.tweaker import Tweaker
from src.utils import *

from torchvision.utils import save_image

class WarpModel(torch.nn.Module):
    """
    filter model output to only keep the target
    """
    def __init__(self, model, attr_dim):
        super(WarpModel, self).__init__()
        self.model = model
        self.attr_dim = attr_dim

    def forward(self, x):
        output = self.model(x)
        return output[:,self.attr_dim]

def main(args):
    print('Pytorch is running on version: ' + torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_time = time.perf_counter()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # resolve the directory
    fig_root = Path(args.out_dir)
    fig_root.mkdir(parents=True, exist_ok=True)

    if args.adv_type != "clean":
        tweaker = Tweaker(batch_size=args.batch_size, tweak_type=args.adv_type, device="cpu")
    # load the element
    element_path = Path(args.stats)
    adv_component = np.load(element_path)
    adv_component = torch.from_numpy(adv_component)

    # load every thing we need
    match args.dataset:
        case "CelebA":
            # Get dataloader and base model
            attr_dim = args.attr_list.index(args.attr_choose)
            dataset = CelebA(batch_size=args.batch_size, attr_list=args.attr_list, 
                            with_xfrom=True, root='/tmp2/dataset/celeba_tm')
            dataloader = dataset.val_dataloader
            attr_count = len(args.attr_list)
            model = BinaryModel(out_feature=attr_count, weights=None).to(device)
            # prepare batch of image, target, and normalized image
            data, raw_label, theta = next(iter(dataloader))
            theta = theta.to(torch.float32)
            # add tweak if needed
            match args.adv_type:
                case "noise" | "patch" | "frame":
                    data, raw_label = tweaker.apply(data, raw_label, adv_component)
                case "eyeglasses":
                    data, raw_label = tweaker.apply(data, raw_label, adv_component, theta)
            # get rgb_image (for showing), GradCam target, and input tensor (for model input) into list
            rgb_images, targets, input_tensor = list(), list(), normalize(data).to(device)
            model.eval()
            model_output = model(input_tensor)
            model_prediction = torch.where(model_output > 0.5, 1, 0).cpu()
            for idx in range(args.batch_size):
                rbg_image = data[idx].permute(1, 2, 0).numpy()
                rgb_images.append(rbg_image)
                # target = BinaryClassifierOutputTarget(raw_label[idx, attr_dim])
                target = BinaryClassifierOutputTarget(model_prediction[idx, attr_dim])
                targets.append(target)
        case "FairFaceAge":
            attr_dim = args.attr_list.index(args.attr_choose)
            dataset = FairFace(batch_size=args.batch_size, with_xfrom=True,
                                train_csv='/tmp2/dataset/fairface-img-margin025-trainval/fairface_label_tm_train.csv',
                                val_csv='/tmp2/dataset/fairface-img-margin025-trainval/fairface_label_tm_val.csv')
            dataloader = dataset.val_dataloader
            attr_count = len(args.attr_list)
            model = BinaryModel(out_feature=attr_count, weights=None).to(device)
            # prepare batch of image, target, and normalized image
            data, raw_label, theta = next(iter(dataloader))
            theta = theta.to(torch.float32)
            match args.adv_type:
                case "noise" | "patch" | "frame":
                    data, raw_label = tweaker.apply(data, raw_label, adv_component)
                case "eyeglasses":
                    data, raw_label = tweaker.apply(data, raw_label, adv_component, theta)
            rgb_images, targets, input_tensor = list(), list(), normalize(data).to(device)
            model.eval()
            model_output = model(input_tensor)
            model_prediction = torch.where(model_output > 0.5, 1, 0).cpu()
            for idx in range(args.batch_size):
                rbg_image = data[idx].permute(1, 2, 0).numpy()
                rgb_images.append(rbg_image)
                # target = BinaryClassifierOutputTarget(raw_label[idx, attr_dim])
                target = BinaryClassifierOutputTarget(model_prediction[idx, attr_dim])
                targets.append(target)
        case "HAM10000":
            attr_dim = [i for i in range(7)] # only skin diagnosis
            dataset = HAM10000(batch_size=args.batch_size)
            dataloader = dataset.val_dataloader
            assert args.adv_type != "eyeglasses", "HAM10000 dataset has no faces"
            model = CategoricalModel(out_feature=7, weights=None).to(device)
            data, raw_label = next(iter(dataloader))
            match args.adv_type:
                case "noise" | "patch" | "frame":
                    data, raw_label = tweaker.apply(data, raw_label, adv_component)
            rgb_images, targets, input_tensor = list(), list(), normalize(data).to(device)
            # TODO: model prediction
            for idx in range(args.batch_size):
                rbg_image = data[idx].permute(1, 2, 0).numpy()
                rgb_images.append(rbg_image)
                # TODO: change target into predictions
                target = ClassifierOutputTarget(raw_label[idx,])
                targets.append(target)
        case _:
            assert False, "Only CelebA, FairFace, and HAM10000 are supported"
    # load checkpoints and warp the model to filter its outputs
    model_ckpt_path = Path(args.model_ckpt_root) / args.model_name / (args.model_ckpt_name+".pth")
    ckpt = torch.load(model_ckpt_path)
    model.load_state_dict(ckpt['model_state_dict'])
    warp_model = WarpModel(model, attr_dim)
    # warp model - binary(categori) model -resnet34
    target_layers = [warp_model.model.model.layer4[-1]]
    cam = GradCAM(model=warp_model, target_layers=target_layers, use_cuda=True)
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets) # shape (N, H, W) float 32
    

    for idx in range(args.batch_size):
        visualization = show_cam_on_image(rgb_images[idx], grayscale_cam[idx], use_rgb=True)
        
        fig_path = fig_root / (args.name + f'_{idx:04d}.png')
        im = Image.fromarray(visualization)
        im.save(fig_path)
        # print(f"Image {idx:2d} label {targets[idx].category.item()}, prediction {model_output[idx, attr_dim].item():.2f}")
        
def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Grad-cam for tweaks")
    # For base model loaded
    parser.add_argument("--seed", default=32138, type=int, help="seed for the adversarial instance training process")
    parser.add_argument("-b", "--batch-size", default=128, type=int, help="batch size for model inputs")
    parser.add_argument("--dataset", type=str, help="the dataset to extract image from")
    parser.add_argument("--model-ckpt-root", default='/tmp2/npfe/model_checkpoint', type=str, help='root path for model checkpoint')
    parser.add_argument("--model-name", default='default_model', type=str, help='name for this model trained')
    parser.add_argument("--model-ckpt-name", default='default_model', type=str, help='name for the model checkpoint, without .pth')
    parser.add_argument("--attr-list", type=str, nargs='+', help="attributes name predicted by model")
    parser.add_argument("--attr-choose", type=str, help="attributes name for grad-cam, only 1 of it")

    # For input tweaking element
    parser.add_argument("--stats", type=str, help="file path to the element .npy file")
    parser.add_argument("--adv-type", default=None, type=str, help="type of adversarial element, only 'noise', 'patch', 'frame', and 'eyeglasses' are allowed")
    parser.add_argument("-o", "--out-dir", type=str, default="./img", help="output folder for image")
    parser.add_argument("-n", "--name", type=str, help="output image name")
    return parser

if __name__ == '__main__':
    args = get_args().parse_args()
    main(args)