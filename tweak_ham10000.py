import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.datasets import HAM10000
from src.models import CategoricalModel
from src.tweaker import Tweaker, Losses
from src.utils import *

def main(args):
    print('Pytorch is running on version: ' + torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    time_start = time.perf_counter()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # dataset, dataloader (HAM10000)
    ham10000 = HAM10000(batch_size=args.batch_size)
    train_dataloader = ham10000.train_dataloader
    val_dataloader = ham10000.val_dataloader

    # the base model, optimizer, and scheduler
    print(f'Calling model predicting case')
    model = CategoricalModel(out_feature=7, weights=None).to(device)
    _optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-5
    )
    _scheduler = torch.optim.lr_scheduler.StepLR(_optimizer, step_size=30, gamma=0.1)
    model_ckpt_path = Path(args.model_ckpt_root)/args.model_name
    load_model(model, _optimizer, _scheduler, name=args.model_ckpt_name, root_folder=model_ckpt_path)
    
    # tweaking element
    advatk_ckpt_path = Path(args.advatk_ckpt_root)/args.advatk_name
    advatk_stat_path = Path(args.advatk_stat_root)/args.advatk_name
    match args.adv_type:
        case "noise" | "patch":
            adv_component = torch.full((1, 3, 224, 224), 0.0).to(device)
        case "frame" | "eyeglasses":
            adv_component = torch.full((1, 3, 224, 224), 0.5).to(device)
        case _:
            assert False, "Unknown element type"
    tweaker = Tweaker(batch_size=args.batch_size, tweak_type=args.adv_type)
    losses = Losses(loss_type=args.loss_type, fairness_criteria=args.fairness_matrix, 
                    pred_type='categorical', dataset_name='HAM10000', soft_label=False)

    if args.resume:
        adv_component = load_stats(name=args.resume, root_folder=advatk_ckpt_path)
        adv_component = torch.from_numpy(adv_component).to(device)
        train_stat = load_stats(name='train', root_folder=advatk_stat_path)
        val_stat = load_stats(name='val', root_folder=advatk_stat_path)
    else:
        train_stat, val_stat = np.array([]), np.array([])
    adv_component = nn.Parameter(adv_component)
    adversary_optimizer = torch.optim.SGD([adv_component], lr=args.lr, momentum=1e-6)
    adversary_scheduler = torch.optim.lr_scheduler.StepLR(adversary_optimizer, step_size=1, gamma=0.9)
    coef = torch.tensor(args.coef).to(device) 
    total_time = time.perf_counter() - time_start
    print(f'Preparation done in {total_time:.4f} secs')

    def to_prediction(logit):
        _, case_pred = torch.max(logit[:,0:7], dim=1)
        return case_pred.unsqueeze(1)
    
    # train and validation function
    def train():
        train_stat = np.array([])
        model.eval()
        # training loop
        for batch_idx, (data, raw_label) in enumerate(train_dataloader):
            data, raw_label = data.to(device), raw_label.to(device)
            # tweak on data
            data, raw_label = tweaker.apply(data, raw_label, adv_component)
            label, sens = raw_label[:,0:1], raw_label[:,1:2] # label: case, sensitive attribute: gender
            instance = normalize(data)
            adversary_optimizer.zero_grad()
            logit = model(instance)
            loss = losses.run(logit, label, sens, coef) # categorical
            loss.backward()
            # if batch_idx % 128 ==0:
            #     adversary_optimizer.step()
            #     tweaker.retify(adv_component)
            #     adversary_optimizer.zero_grad()
            adversary_optimizer.step()
            tweaker.retify(adv_component)
            # collecting performance information
            pred = to_prediction(logit)
            stat = calc_groupacc(pred, label, sens)
            stat = stat[np.newaxis, :]
            train_stat = train_stat+stat if len(train_stat) else stat
        return train_stat # in shape (1, ?, 4), ?: race + gender + age, 4: split in 2 groups x (right and wrong)
    
    def val(dataloader=val_dataloader, times=1):
        val_stat = np.array([])
        model.eval()
        with torch.no_grad():
            # validation may run more than one time per epoch
            for _ in range(times):
                # validaton loop
                for batch_idx, (data, raw_label) in enumerate(dataloader):
                    data, raw_label = data.to(device), raw_label.to(device)
                    # tweak on data
                    data, raw_label = tweaker.apply(data, raw_label, adv_component)
                    label, sens = raw_label[:,0:1], raw_label[:,1:2] # label: case, sensitive attribute: gender
                    instance = normalize(data)
                    logit = model(instance)
                    # collecting performance information
                    pred = to_prediction(logit)
                    stat = calc_groupacc(pred, label, sens)
                    stat = stat[np.newaxis, :]
                    val_stat = val_stat+stat if len(val_stat) else stat
            return val_stat # in shape (1, ?, 4), ?: race + gender + age, 4: split in 2 groups x (right and wrong)
    # summarize the status in validation set for some adjustment
    def get_stats_per_epoch(stat):
        # Input: statistics for a single epochs, shape (1, ?, 4)
        group_1_correct, group_1_wrong, group_2_correct, group_2_wrong = [stat[0,:,i] for i in range(0, 4)]
        group_1_acc = group_1_correct/(group_1_correct+group_1_wrong)
        group_2_acc = group_2_correct/(group_2_correct+group_2_wrong)
        total_acc = (group_1_correct+group_2_correct)/(group_1_correct+group_1_wrong+group_2_correct+group_2_wrong)
        acc_diff = abs(group_1_acc-group_2_acc)
        stat_dict = {"group_1_acc": group_1_acc, "group_2_acc": group_2_acc, 
                     "total_acc": total_acc, "acc_diff": acc_diff}
        return stat_dict
    # print the epoch status on to the terminal
    def show_stats_per_epoch(train_stat_per_epoch, val_stat_per_epoch):
        attr_list = ["Case",]
        for index, attr_name in enumerate(attr_list):
            print(f'    attribute: {attr_name: >40}')
            stat_dict = get_stats_per_epoch(train_stat_per_epoch)
            group_1_acc, group_2_acc = stat_dict["group_1_acc"][index], stat_dict["group_2_acc"][index]
            total_acc, acc_diff = stat_dict["total_acc"][index], stat_dict["acc_diff"][index]
            print(f'    train    {group_1_acc:.4f} - {group_2_acc:.4f} - {total_acc:.4f} -- {acc_diff:.4f}')
            stat_dict = get_stats_per_epoch(val_stat_per_epoch)
            group_1_acc, group_2_acc = stat_dict["group_1_acc"][index], stat_dict["group_2_acc"][index]
            total_acc, acc_diff = stat_dict["total_acc"][index], stat_dict["acc_diff"][index]
            print(f'    val      {group_1_acc:.4f} - {group_2_acc:.4f} - {total_acc:.4f} -- {acc_diff:.4f}')
        print(f'')

    # Run the code
    print(f'Start training model')
    time_start = time.perf_counter()

    if not args.resume:
        empty_time = time.time()
        print(f'collecting statistic for empty tweaks')
        train_stat_per_epoch = val(train_dataloader, times=1)
        val_stat_per_epoch = val(times=11)
        train_stat = np.concatenate((train_stat, train_stat_per_epoch), axis=0) if len(train_stat) else train_stat_per_epoch
        val_stat = np.concatenate((val_stat, val_stat_per_epoch), axis=0) if len(val_stat) else val_stat_per_epoch
        show_stats_per_epoch(train_stat_per_epoch, val_stat_per_epoch)
        print(f'done in {(time.time()-empty_time)/60:.4f} mins')
    # some parameter might needs the init stats

    for epoch in range(args.start_epoch, args.epochs):
        epoch_start = time.time()
        train_stat_per_epoch = train()
        # scheduler.step()
        val_stat_per_epoch = val(times=11)
        epoch_time = time.time() - epoch_start
        print(f'Epoch {epoch:4} done in {epoch_time/60:.4f} mins')
        train_stat = np.concatenate((train_stat, train_stat_per_epoch), axis=0) if len(train_stat) else train_stat_per_epoch
        val_stat = np.concatenate((val_stat, val_stat_per_epoch), axis=0) if len(val_stat) else val_stat_per_epoch
        # adjust the recovery loss coefficient
        if args.coef_mode == "dynamic":
            init_tacc = get_stats_per_epoch(val_stat[0:1,:,:])["total_acc"]
            last_tacc = get_stats_per_epoch(val_stat[-2:-1,:,:])["total_acc"]
            curr_tacc = get_stats_per_epoch(val_stat_per_epoch)["total_acc"]
            fairness_dict_key = fairness_matrix_to_dict_key(args.fairness_matrix)
            last_fairness = get_stats_per_epoch(val_stat[-2:-1,:,:])[fairness_dict_key]
            curr_fairness = get_stats_per_epoch(val_stat_per_epoch)[fairness_dict_key]
            # print the coefficient (categorical)
            coef_list = coef.clone().cpu().numpy().tolist()
            coef_list = [f'{f:.4f}' for f in coef_list]
            print(f'    coef: {" ".join(coef_list)}')
            if curr_tacc[0] < init_tacc[0] - args.quality_target: # 0: Case
                coef[0] = min(coef[0]*1.05, 1e4)
            elif curr_tacc[0] > args.fairness_target and curr_fairness[0] > last_fairness[0]:
                coef[0] = max(coef[0]*0.95, 1e-4)
        # print some basic statistic
        show_stats_per_epoch(train_stat_per_epoch, val_stat_per_epoch)
        # save the adversarial component for each epoch
        component = adv_component.detach().cpu().numpy()
        save_stats(component, f'{epoch:04d}', root_folder=advatk_ckpt_path)
    # save basic statistic
    save_stats(train_stat, f'train', root_folder=advatk_stat_path)
    save_stats(val_stat, f'val', root_folder=advatk_stat_path)
    total_time = time.perf_counter() - time_start
    print(f'Training time: {total_time/60:.4f} mins')

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Tweak the model by data pre-processing")
    # For base model loaded
    parser.add_argument("--model-ckpt-root", default='/tmp2/npfe/model_checkpoint', type=str, help='root path for model checkpoint')
    # parser.add_argument("--model-stat-root", default='/tmp2/npfe/model_stats', type=str, help='root path for model statistic')
    parser.add_argument("--model-name", default='default_model', type=str, help='name for this model trained')
    parser.add_argument("--model-ckpt-name", default='default_model', type=str, help='name for the model checkpoint, without .pth')
    parser.add_argument("--attr-list", type=str, nargs='+', help="attributes name predicted by model")

    # For input tweaking element
    parser.add_argument("--advatk-ckpt-root", default='/tmp2/npfe/advatk', type=str, help='root path for adversarial atttack statistic')
    parser.add_argument("--advatk-stat-root", default='/tmp2/npfe/advatk_stats', type=str, help='root path for adversarial attack itself')
    parser.add_argument("--advatk-name", default='default_advatk', type=str, help='name for the advatk trained')
    parser.add_argument("--resume", default="", help="name of a adversarial element, without .npy")
    parser.add_argument("--start-epoch", default=0, type=int, help="start epoch, it won't do any check on the element loaded")
    # training related
    parser.add_argument("--seed", default=32138, type=int, help="seed for the adversarial instance training process")
    parser.add_argument("-b", "--batch-size", default=128, type=int, help="batch size for model inputs")
    parser.add_argument("--epochs", default=125, type=int, help="number of epochs to run")
    parser.add_argument("--lr", default=1e-1, type=float, help="step size for model training")
    parser.add_argument("--adv-type", default=None, type=str, help="type of adversarial element, only 'noise', 'patch', 'frame', and 'eyeglasses' are allowed")
    # setting for each types of tweek



    parser.add_argument("--fairness-matrix", default="prediction quaility", help="how to measure fairness")
    # binary model
    # parser.add_argument("--p-coef", default=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1,], type=float, nargs='+', help="coefficient multiply on positive recovery loss, need to be match with the number of attributes")
    # parser.add_argument("--n-coef", default=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1,], type=float, nargs='+', help="coefficient multiply on negative recovery loss, need to be match with the number of attributes")
    # categorical model 
    parser.add_argument("--coef", default=[0.0,], type=float, nargs='+', help="coefficient multiply on negative recovery loss, need to be match with the number of attributes")
    # loss types
    parser.add_argument("--loss-type", default='direct', type=str, help="Type of loss used")
    parser.add_argument("--coef-mode", default="static", type=str, help="method to adjust coef durinig training")
    parser.add_argument("--fairness-target", default=0.03, type=float, help="Fairness target value")
    parser.add_argument("--quality-target", default=0.05, type=float, help="Max gap loss for prediction quaility")

    return parser

if __name__ == '__main__':
    args = get_args().parse_args()
    main(args)