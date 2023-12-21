import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.datasets import CelebA
from src.models import BinaryModel
from src.utils import *

def main(args):
    print('Pytorch is running on version: ' + torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_time = time.perf_counter()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # dataset, dataloader (CelebA)
    attr_list = ['Attractive', 'Eyeglasses']
    celeba = CelebA(batch_size=1, attr_list=attr_list)
    train_dataloader = celeba.train_dataloader
    val_dataloader = celeba.val_dataloader

    count = [0, 0, 0, 0]
    # count the validation set
    for batch_idx, (data, raw_label) in enumerate(val_dataloader):
        attractive, eyeglasses = raw_label[0,0], raw_label[0,1]
        match (attractive.item(), eyeglasses.item()):
            case (0, 0):
                count[0] += 1
            case (0, 1):
                count[1] += 1
            case (1, 0):
                count[2] += 1
            case (1, 1):
                count[3] += 1
    print(count)


def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="")
    # Training related arguments
    parser.add_argument("--seed", default=32138, type=int, help="seed for the model training process")
    parser.add_argument("-b", "--batch-size", default=128, type=int, help="batch size for model inputs")
    parser.add_argument("--epochs", default=125, type=int, help="number of epochs to run")
    parser.add_argument("--lr", default=1e-1, type=float, help="step size for model training")
    # For model trained
    parser.add_argument("--model-ckpt-root", default='/tmp2/npfe/model_checkpoint', type=str, help='root path for model checkpoint')
    parser.add_argument("--model-stat-root", default='/tmp2/npfe/model_stats', type=str, help='root path for model statistic')
    parser.add_argument("--model-name", default='default_model', type=str, help='name for this model trained')
    parser.add_argument("--resume", default="", help="name of a checkpoint, without .pth")
    parser.add_argument("--start-epoch", default=0, type=int, help="start epoch, it won't do any check with the weight loaded")
    parser.add_argument("--attr-list", type=str, nargs='+', help="attributes name predicted by model")

    return parser


if __name__ == '__main__':
    args = get_args().parse_args()
    main(args)