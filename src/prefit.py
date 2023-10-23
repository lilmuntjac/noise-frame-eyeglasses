from pathlib import Path
import numpy as np
import pandas as pd
import random
import time
import dlib

import torch

def main(args):

    width = 224
    # model from dlib
    face_detector: dlib.cnn_face_detection_model_v1 = \
        dlib.cnn_face_detection_model_v1('../dlib_models/mmod_human_face_detector.dat')
    shape_predictor: dlib.shape_predictor = \
        dlib.shape_predictor('../dlib_models/shape_predictor_68_face_landmarks.dat')
    # transformation matrix related
    coord_ref = np.linspace(-3.0, 3.0, num=6*width+1)
    coord_ref = coord_ref[1::2] # remove the odd index (edge between pixels)
    coord_ref = np.around(coord_ref, 4)
    # note: true_coord_ref = coord_ref[pixel_coord+width]
    def get_torch_coord(point_list):
        width=224
        new_coord = list()
        for point in point_list:
            x, y = int(point[0]), int(point[1])
            # landmark upper bound & lower bound
            new_x, new_y = coord_ref[x+width], coord_ref[y+width]
            new_coord.append([new_x, new_y])
        return new_coord
    reference_pixel = [[73,75],[149,75],[111,130]] # on eyeglasses mask
    reference = get_torch_coord(reference_pixel)

    def fit_celeba(out_folder, root_folder='/tmp2/dataset/celeba'):
        """
        read celeba dataset, detect face with dlib, solve the transformation matrix for eyeglasses
        save the image file name and transformation matrix into csv file
        """
        # read the original csv
        attr_header = ['filename', '5_o_Clock_Shadow','Arched_Eyebrows','Attractive','Bags_Under_Eyes',
                       'Bald','Bangs','Big_Lips','Big_Nose','Black_Hair','Blond_Hair','Blurry',
                       'Brown_Hair','Bushy_Eyebrows','Chubby','Double_Chin','Eyeglasses','Goatee',
                       'Gray_Hair','Heavy_Makeup','High_Cheekbones','Male','Mouth_Slightly_Open',
                       'Mustache','Narrow_Eyes','No_Beard','Oval_Face','Pale_Skin','Pointy_Nose',
                       'Receding_Hairline','Rosy_Cheeks','Sideburns','Smiling','Straight_Hair',
                       'Wavy_Hair','Wearing_Earrings','Wearing_Hat','Wearing_Lipstick','Wearing_Necklace',
                       'Wearing_Necktie','Young']
        root_folder = Path(root_folder) 
        splits = pd.read_csv(root_folder / "list_eval_partition.txt", sep="\s+", names=['filename', 'split'])
        attr   = pd.read_csv(root_folder / "list_attr_celeba.txt", sep="\s+", skiprows=2, names = attr_header,)
        attr_splits = pd.merge(attr, splits, on='filename')
        # output csv
        out_header = attr_header.copy() + ['split', 'a11', 'a12', 'a13', 'a21', 'a22', 'a23']
        out_folder = Path(out_folder)
        df_all = pd.DataFrame(columns=out_header)

        for index in attr_splits.index:
            image_path = root_folder / 'img_align_celeba' / attr_splits['filename'][index]
            image = dlib.load_rgb_image(str(image_path)) # numpy asrray [H, W, 3] unit8
            image = dlib.resize_image(image, rows=224, cols=224)
            detected_face = face_detector(image, 1)
            if len(detected_face) != 1:
                continue # more than 1 face, discard this image
            landmark = shape_predictor(image, detected_face[0].rect)
            left_eye = (landmark.part(36)+landmark.part(39))/2
            right_eye = (landmark.part(42)+landmark.part(45))/2
            noise_tip = landmark.part(33)
            destination = [[left_eye.x, left_eye.y], 
                           [right_eye.x, right_eye.y], 
                           [noise_tip.x, noise_tip.y]]
            destination = get_torch_coord(destination)
            for point in destination:
                point.append(1)
            xform_matrix = np.linalg.solve(destination, reference)
            xform_matrix = np.transpose(xform_matrix)
            row = attr_splits.iloc[index].values.flatten().tolist() + \
                  [f'{xform_matrix[0,0]:.5f}', f'{xform_matrix[0,1]:.5f}', f'{xform_matrix[0,2]:.5f}',
                   f'{xform_matrix[1,0]:.5f}', f'{xform_matrix[1,1]:.5f}', f'{xform_matrix[1,2]:.5f}']
            df_all.loc[len(df_all.index)] = row
            if index > 0 and index % 5000 == 0:
                print(f'{index:7d} image computed.')

        # split the csv into 3
        attr_train = df_all.loc[df_all["split"] == 0, out_header]
        attr_train = attr_train.drop('split', axis=1)
        attr_train.to_csv(out_folder / 'celeba_tm_train.csv', index=False)
        attr_val = df_all.loc[df_all["split"] == 1, out_header]
        attr_val = attr_val.drop('split', axis=1)
        attr_val.to_csv(out_folder / 'celeba_tm_val.csv', index=False)
        attr_test = df_all.loc[df_all["split"] == 2, out_header]
        attr_test = attr_test.drop('split', axis=1)
        attr_test.to_csv(out_folder / 'celeba_tm_test.csv', index=False)

    def fit_fairface(out_folder, root_folder='/tmp2/dataset/celeba'):
        """
        read fairface dataset, detect face with dlib, solve the transformation matrix for eyeglasses
        save the image file name and transformation matrix into csv file
        """

        root_folder = Path(root_folder)
        out_folder = Path(out_folder)

        def fit(csv_in):
            df_out = pd.DataFrame(columns=['file', 'age', 'gender', 'race', 'service_test',
                                           'a11', 'a12', 'a13', 'a21', 'a22', 'a23'])
            # read csv
            csv = pd.read_csv(root_folder / csv_in)
            for index in csv.index:
                image_path = root_folder / csv.iloc[index]['file']
                image = dlib.load_rgb_image(str(image_path)) # numpy asrray [H, W, 3] unit8
                image = dlib.resize_image(image, rows=224, cols=224)
                detected_face = face_detector(image, 1)
                if len(detected_face) != 1:
                    continue # more than 1 face, discard this image
                landmark = shape_predictor(image, detected_face[0].rect)
                left_eye = (landmark.part(36)+landmark.part(39))/2
                right_eye = (landmark.part(42)+landmark.part(45))/2
                noise_tip = landmark.part(33)
                destination = [[left_eye.x, left_eye.y], 
                               [right_eye.x, right_eye.y], 
                               [noise_tip.x, noise_tip.y]]
                destination = get_torch_coord(destination)
                for point in destination:
                    point.append(1)
                xform_matrix = np.linalg.solve(destination, reference)
                xform_matrix = np.transpose(xform_matrix)
                row = csv.iloc[index].values.flatten().tolist() + \
                      [f'{xform_matrix[0,0]:.5f}', f'{xform_matrix[0,1]:.5f}', f'{xform_matrix[0,2]:.5f}',
                       f'{xform_matrix[1,0]:.5f}', f'{xform_matrix[1,1]:.5f}', f'{xform_matrix[1,2]:.5f}']
                df_out.loc[len(df_out.index)] = row
                if index > 0 and index % 5000 == 0:
                    print(f'{index:7d} image computed.')
            return df_out

        # read the original csv
        train_df = fit(root_folder / "fairface_label_train.csv")
        val_df = fit(root_folder / "fairface_label_val.csv")
        train_df.to_csv(out_folder / "fairface_label_tm_train.csv", index=False)
        val_df.to_csv(out_folder / "fairface_label_tm_val.csv", index=False)
    
    def fit_utkface():
        pass

    time_start = time.perf_counter()

    match args.dataset:
        case "CelebA":
            fit_celeba(args.out_folder, args.root_folder)
        case "FairFace":
            fit_fairface(args.out_folder, args.root_folder)

    time_end = time.perf_counter()
    print(f'done in {(time_end-time_start)/60:.4f} mins')

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Split the dataset into training and validation")
    parser.add_argument("--dataset", type=str, default=None, help="name of the dataset to split")
    parser.add_argument("--root-folder", type=str)
    parser.add_argument("--root-csv", type=str)
    parser.add_argument("--out-folder", type=str)

    return parser

if __name__ == '__main__':
    args = get_args().parse_args()
    main(args)