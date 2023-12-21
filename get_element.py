from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from torchvision.utils import save_image

from src.datasets import CelebA, FairFace, HAM10000
from src.tweaker import Tweaker, Losses
from src.utils import *

def main(args):
    print('Pytorch is running on version: ' + torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    tweaker = Tweaker(batch_size=args.batch_size, tweak_type=args.adv_type)
    # load the element
    element_path = Path(args.stats)
    adv_component = np.load(element_path)
    adv_component = torch.from_numpy(adv_component).to(device)
    print(f"The element is in shape of: {adv_component.shape}")
    # resolve the directory
    fig_root = Path(args.out_dir)
    fig_root.mkdir(parents=True, exist_ok=True)

    # dataset and dataloader
    match args.dataset:
        case "CelebA":
            dataset = CelebA(batch_size=args.batch_size, 
                            with_xfrom=True, root='/tmp2/dataset/celeba_tm')
            dataloader = dataset.val_dataloader
            data, raw_label, theta = next(iter(dataloader))
        case "FairFaceAge":
            dataset = FairFace(batch_size=args.batch_size, with_xfrom=True,
                                train_csv='/tmp2/dataset/fairface-img-margin025-trainval/fairface_label_tm_train.csv',
                                val_csv='/tmp2/dataset/fairface-img-margin025-trainval/fairface_label_tm_val.csv')
            dataloader = dataset.val_dataloader
            data, raw_label, theta = next(iter(dataloader))
        case "HAM10000":
            dataset = HAM10000(batch_size=args.batch_size)
            dataloader = dataset.val_dataloader
            data, raw_label = next(iter(dataloader))
            assert args.adv_type != "eyeglasses", "HAM10000 dataset has no faces"
        case _:
            assert False, "Only CelebA, FairFace, and HAM10000 are supported"

    # black_img = torch.zeros((args.batch_size, 3, 224, 224)).to(device)
    gray_img = torch.full((args.batch_size, 3, 224, 224), 0.5).to(device)
    dummy_label = torch.zeros((args.batch_size, 1)).to(device)

    match args.show_type:
        case "element":
            match args.adv_type:
                case "noise":
                    # enhance the noise to make it more obvious
                    adv_component*=8
                    element, _ = tweaker.apply(gray_img, dummy_label, adv_component)
                case "patch" | "frame" | "eyeglasses":
                    # save the raw image as-is
                    element = adv_component
            # only need to save the single image
            fig_path = fig_root / (args.name + '.png')
            save_image(element[0], fig_path)
        case "deploy":
            data = data.to(device)
            if args.adv_type == "eyeglasses":
                element, _ = tweaker.apply(data, raw_label, adv_component, theta.to(torch.float32).to(device))
            else:
                element, _ = tweaker.apply(data, raw_label, adv_component)
            # save the entire batch
            for i in range(element.shape[0]):
                fig_path = fig_root / (args.name + f'_{i:04d}.png')
                save_image(element[i], fig_path)
        case "clean":
            # adv type should be noise
            data = data.to(device)
            empty = torch.full((args.batch_size, 3, 224, 224), 0.0).to(device)
            element, _ = tweaker.apply(data, raw_label, empty)
            # save the entire batch
            for i in range(element.shape[0]):
                fig_path = fig_root / (args.name + f'_{i:04d}.png')
                save_image(element[i], fig_path)
        case _:
            assert False, "unknown show type"
    
def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="output the tweak element as image")
    # For input tweaking element
    parser.add_argument("--dataset", type=str, help="the dataset to extract image from")
    parser.add_argument("--adv-type", default=None, type=str, help="type of adversarial element, only 'noise', 'patch', 'frame', and 'eyeglasses' are allowed")
    parser.add_argument("--stats", type=str, help="file path to the element .npy file")
    parser.add_argument("--seed", default=32138, type=int, help="seed for the adversarial instance training process")
    parser.add_argument("-b", "--batch-size", default=128, type=int, help="batch size for model inputs")
    parser.add_argument("-o", "--out-dir", type=str, default="./img", help="output folder for image")
    parser.add_argument("-n", "--name", type=str, help="output image name")

    parser.add_argument("--show-type", default="element" ,type=str, help="Get the element itself or the element apply onto the gray image. 'element' or 'deploy'")

    return parser

if __name__ == '__main__':
    args = get_args().parse_args()
    main(args)