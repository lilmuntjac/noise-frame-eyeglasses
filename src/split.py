from pathlib import Path
import numpy as np
import pandas as pd
import random
import time

def main(args):

    def UTKFace_split(root_folder='/tmp2/dataset/UTKFace', seed=14798):
        """
        Split the UTKFace dataset into train and validation set
        Given a folder, find path for all the image to create training and validation csv
        """
        root_dir = Path(root_folder)
        image_list = list((root_dir).glob('**/[0-9]*_[0-9]*_[0-9]*_[0-9]*.jpg.chip.jpg'))
        image_list.sort()
        random.Random(seed).shuffle(image_list)
        # split the list into 8:2
        train, val = image_list[:int(len(image_list)*0.8)], image_list[int(len(image_list)*0.8):]
        train_df = pd.DataFrame(columns=['file', 'age', 'gender', 'race'])
        for p in train:
            p_stem = str(p.stem).split('_')
            age, gender, race = p_stem[0], p_stem[1], p_stem[2]
            train_df.loc[len(train_df.index)] = [p.name, age, gender, race]
        val_df = pd.DataFrame(columns=['file', 'age', 'gender', 'race'])
        for p in val:
            p_stem = str(p.stem).split('_')
            age, gender, race = p_stem[0], p_stem[1], p_stem[2]
            val_df.loc[len(val_df.index)] = [p.name, age, gender, race]
        train_df.to_csv(root_dir/'utkface_train.csv', index=False)
        val_df.to_csv(root_dir/'utkface_val.csv', index=False)

    def HAM10000_split(root_csv, root_folder='/tmp2/dataset/HAM10000' ,seed=14798):
        """
        Split the HAM10000 original training set (10015 of them)
        into training and validation dataset, the original test set has poor label quality
        root_scv should be the path to the original csv of training label 
        """
        root_dir = Path(root_folder)
        rng = np.random.RandomState(seed)
        original_csv = pd.read_csv(root_csv)
        train_df = original_csv.sample(frac=0.8, random_state=rng)
        val_df = original_csv.loc[~original_csv.index.isin(train_df.index)]
        train_df.to_csv(root_dir/'ham10000_train.csv', index=False)
        val_df.to_csv(root_dir/'ham10000_val.csv', index=False)

    time_start = time.perf_counter()
    match args.dataset:
        case "UTKFace":
            UTKFace_split(args.root_folder, args.seed)
        case "HAM10000":
            HAM10000_split(args.root_csv, args.root_folder, args.seed)
        case _:
            assert False, "Unknown dataset"

    time_end = time.perf_counter()
    print(f'done in {(time_end-time_start)/60:.4f} mins')


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Split the dataset into training and validation")
    parser.add_argument("--dataset", type=str, default=None, help="name of the dataset to split")
    parser.add_argument("--root-folder", type=str)
    parser.add_argument("--root-csv", type=str)
    parser.add_argument("-s", "--seed", type=int)

    return parser

if __name__ == '__main__':
    args = get_args().parse_args()
    main(args)

