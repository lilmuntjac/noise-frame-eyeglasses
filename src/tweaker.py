from dataclasses import dataclass, field
from typing import Tuple
from collections import OrderedDict
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import dlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from .perturbations import *
from .utils import *

@dataclass
class Tweaker:
    tweak_type: str

    batch_size: int = 128
    width: int = 224
    device: str = 'cuda'

    noise_budget: float = 0.0625
    circle_sharpness: float = 40.
    circle_rotation: float = 0.083
    circle_ratio: tuple[float, float] = (0.02, 0.03)
    frame_thickness: float = 0.25
    provide_theta: bool = True
    face_detector: dlib.cnn_face_detection_model_v1 = \
        dlib.cnn_face_detection_model_v1('./dlib_models/mmod_human_face_detector.dat')
    shape_predictor: dlib.shape_predictor = \
        dlib.shape_predictor('./dlib_models/shape_predictor_68_face_landmarks.dat')

    # create after the width is defined
    coord_ref: list[float] = field(default_factory=list, init=False)
    # create after the tweak type is defined
    mask: torch.Tensor = field(init=False)
    def __post_init__(self):
        # init coord_ref
        coord_ref = np.linspace(-3.0, 3.0, num=6*self.width+1)
        coord_ref = coord_ref[1::2] # remove the odd index (edge between pixels)
        coord_ref = np.around(coord_ref, 4)
        # note: true_coord_ref = coord_ref[pixel_coord+width]
        self.coord_ref = coord_ref

        # init mask
        self.mask = self.get_mask()
        self.mask = self.mask.to(self.device)
        
    def get_mask(self):
        match self.tweak_type:
            case 'noise':
                mask = torch.ones(1, 3, self.width, self.width) # don't need a mask
            case 'patch':
                diameter = self.width
                x = torch.linspace(-1, 1, diameter)
                y = torch.linspace(-1, 1, diameter)
                xx, yy = torch.meshgrid(x, y, indexing='ij')
                z = (xx**2 + yy**2) ** self.circle_sharpness
                mask = 1 - np.clip(z, -1, 1)
                mask = mask.unsqueeze(0)
                mask = torch.cat((mask, mask, mask), 0)
                mask = mask.unsqueeze(0)
            case 'frame':
                base = torch.zeros(self.width, self.width)
                gap = self.frame_thickness*self.width/2
                for i in range(self.width):
                    for j in range(self.width):
                        if i < gap or j < gap:
                            base[i][j] = 1
                        if self.width-i < gap or self.width-j < gap:
                            base[i][j] = 1
                mask = base.unsqueeze(0)
                mask = torch.cat((mask, mask, mask), 0)
                mask  = mask.unsqueeze(0)
            case 'eyeglasses':
                image = Image.open('./eyeglasses_mask_6percent.png')
                PIL_to_tensor = transforms.ToTensor()
                mask = PIL_to_tensor(image)
                mask = mask.unsqueeze(0)
            case _:
                assert False, 'the tweak type is not supported.'
        return mask
        
    # -------------------- noise --------------------
    def add(self, x, element):
        x = x + element
        x = torch.clamp(x, min=0.0, max=1.0) # valid image range
        return x
    
    # -------------------- patch --------------------
    def get_circle_transform(self):
        theta = torch.empty(0)
        # create one transformation matrix at a time
        for b in range(self.batch_size):
            # rotation and scaling
            rot = (-2*self.circle_rotation)*torch.rand(1) + self.circle_rotation
            rot_matrix = torch.tensor(
                [[torch.cos(-rot), -torch.sin(-rot)],
                 [torch.sin(-rot), torch.cos(-rot)]]
            )
            scale = map(lambda x : 2*np.sqrt(x/np.pi), self.circle_ratio)
            scale_min, scale_max = scale
            scale = (scale_min-scale_max)*torch.rand(1) + scale_max
            inv_scale = 1.0 / scale
            scale_matrix = torch.tensor(
                [[inv_scale, 0],
                 [0, inv_scale]]
            )
            xform_matrix = torch.mm(rot_matrix, scale_matrix)
            # translation
            avoid_from_center = 0.5
            range_min, range_max = avoid_from_center+scale, 1-scale
            if range_min >= range_max:
                print(f'range min: {range_min}, range max: {range_max}')
                assert False, f'Patch is too large (or too close) to avoid the center of the image.'
            # keep trying until it fit
            while True:
                rnd_min, rnd_max = -(1-scale), 1-scale
                shift_x, shift_y = (rnd_min-rnd_max)*torch.rand(2) + rnd_max
                if abs(shift_x) >= range_min or abs(shift_y) >= range_min:
                    break
            shift_x, shift_y = shift_x*inv_scale, shift_y*inv_scale
            # if scale <= 1.0:
            #     shift_min, shift_max = -(1-scale)/scale, (1-scale)/scale
            # else:
            #     shift_min, shift_max = 0.0, 0.0
            # shift_x, shift_y = (shift_min-shift_max)*torch.rand(2) + shift_max
            xform_matrix = torch.cat((xform_matrix, torch.tensor([[shift_x], [shift_y]])), 1)
            xform_matrix = xform_matrix.unsqueeze(0)
            theta = torch.cat((theta, xform_matrix), dim=0) if len(theta) else xform_matrix
        return theta.to(self.device)
    
    # -------------------- frame --------------------
    def get_identity_transform(self, batch_size):
        theta = torch.tensor([[[1.,0.,0.],[0.,1.,0.]]])
        theta = theta.repeat(batch_size, 1, 1)
        return theta.to(self.device)
    
    # -------------------- eyeglesses --------------------
    def get_landmark(self, data, label):
        # turn the image back to [0, 255] for face detector
        rgb_image = data.clone().detach().cpu().permute(0, 2, 3, 1).numpy()*255.9999
        rgb_image = rgb_image.astype(np.uint8)
        detectable, landmark = list(), list()
        for idx in range(self.batch_size):
            detected_face = self.face_detector(rgb_image[idx], 1)
            if len(detected_face) != 1:
                continue # only 1 face is allowed
            landmark.append(self.shape_predictor(rgb_image[idx], detected_face[0].rect))
            detectable.append(idx)
        filtered_data = data[detectable, :, :, :]
        filtered_label = label[detectable,]
        return filtered_data, filtered_label, landmark
    
    def get_torch_coord(self, point_list):
        new_coord = list()
        for point in point_list:
            x, y = int(point[0]), int(point[1])
            # landmark upper bound & lower bound
            new_x, new_y = self.coord_ref[x+self.width], self.coord_ref[y+self.width]
            new_coord.append([new_x, new_y])
        return new_coord

    def set_eyeglasses_transform(self, landmark, reference=[[73,75],[149,75],[111,130]]):
        reference = self.get_torch_coord(reference)
        theta = torch.empty(0)
        for lm in landmark:
            # get the transformed points from the landmark
            left_eye, right_eye, noise_tip = (lm.part(36)+lm.part(39))/2 ,(lm.part(42)+lm.part(45))/2, lm.part(33)
            destination = [[left_eye.x, left_eye.y], [right_eye.x, right_eye.y], [noise_tip.x, noise_tip.y]]
            destination = self.get_torch_coord(destination)
            for point in destination:
                point.append(1)
            destination = torch.tensor(destination, dtype=torch.float)
            outset = torch.tensor(reference, dtype=torch.float)
            xform_matrix = torch.linalg.solve(destination, outset).transpose(0,1)
            xform_matrix = xform_matrix.unsqueeze(0)
            theta = torch.cat((theta, xform_matrix), dim=0) if len(theta) else xform_matrix
        return theta.to(self.device)

    # --------------------  --------------------
    def apply(self, data, label, element, theta=None):
        match self.tweak_type:
            case 'noise':
                tweak_data = self.add(data, element)
            case 'patch':
                theta = self.get_circle_transform()
                mask = self.mask.repeat(data.shape[0], 1, 1, 1)
                element = element.repeat(data.shape[0], 1, 1, 1)
                grid = F.affine_grid(theta, data.shape, align_corners=False)
                # xform_mask = F.grid_sample(mask, grid, align_corners=False)
                xform_component = F.grid_sample(element, grid, mode='bilinear', align_corners=False)
                # inv_mask = 1 - xform_mask
                # tweak_data = data*inv_mask + xform_component*xform_mask
                # new ver.
                tweak_data = data + xform_component
                tweak_data = torch.clamp(tweak_data, min=0.0, max=1.0)
            case 'frame':
                theta = self.get_identity_transform(batch_size=data.shape[0])
                mask = self.mask.repeat(data.shape[0], 1, 1, 1)
                element = element.repeat(data.shape[0], 1, 1, 1)
                grid = F.affine_grid(theta, data.shape, align_corners=False)
                xform_mask = F.grid_sample(mask, grid, align_corners=False)
                xform_component = F.grid_sample(element, grid, mode='bilinear', align_corners=False)
                inv_mask = 1 - xform_mask
                tweak_data = data*inv_mask + xform_component*xform_mask
            case 'eyeglasses':
                if  self.provide_theta:
                    mask = self.mask.repeat(data.shape[0], 1, 1, 1)
                    element = element.repeat(data.shape[0], 1, 1, 1)
                    grid = F.affine_grid(theta, data.shape, align_corners=False)
                    xform_mask = F.grid_sample(mask, grid, align_corners=False)
                    xform_component = F.grid_sample(element, grid, mode='bilinear', align_corners=False)
                    inv_mask = 1 - xform_mask
                    tweak_data = data*inv_mask + xform_component*xform_mask
                else:
                    # so far only this method remove some data that fail on face detector
                    filtered_data, filtered_label, landmark = self.get_landmark(data, label)
                    theta = self.set_eyeglasses_transform(landmark)
                    mask = self.mask.repeat(filtered_data.shape[0], 1, 1, 1)
                    element = element.repeat(filtered_data.shape[0], 1, 1, 1)
                    grid = F.affine_grid(theta, filtered_data.shape, align_corners=False)
                    xform_mask = F.grid_sample(mask, grid, align_corners=False)
                    xform_component = F.grid_sample(element, grid, mode='bilinear', align_corners=False)
                    inv_mask = 1 - xform_mask
                    tweak_data = filtered_data*inv_mask + xform_component*xform_mask
                    label = filtered_label
            case _:
                assert False, 'the tweak type is not supported.'
        return tweak_data, label
    
    def retify(self, element):
        match self.tweak_type:
            case 'noise':
                element.data.clamp_(-self.noise_budget, self.noise_budget)
            case 'patch' | 'frame' | 'eyeglasses':
                element.data.clamp_(0., 1.)
            case _:
                assert False, 'the tweak type is not supported.'

        
@dataclass
class Losses:
    loss_type: str
    fairness_criteria: str
    pred_type: str
    dataset_name: str
    soft_label: bool = False

    def __post_init__(self):
        if self.pred_type == 'categorical':
            match self.dataset_name:
                case "UTKFace" | "HAM10000" | "FairFace":
                    return
                case _:
                    assert False, "dataset name unsupported"

    # -------------------- cell mask --------------------
    def tp_cells(self, pred, label):
        if self.soft_label:
            label = torch.where(label>0.4, 1, 0) # strong positive label
        return torch.mul(pred, label)
    def fp_cells(self, pred, label):
        if self.soft_label:
            label = torch.where(label>0.4, 1, 0) # strong positive label
        return torch.mul(pred, torch.sub(1, label))
    def fn_cells(self, pred, label):
        if self.soft_label:
            label = torch.where(label<0.6, 0, 1) # strong negative label
        return torch.mul(torch.sub(1, pred), label)
    def tn_cells(self, pred, label):
        if self.soft_label:
            label = torch.where(label<0.6, 0, 1) # strong negative label
        return torch.mul(torch.sub(1, pred), torch.sub(1, label))
    
    # -------------------- cm ratio --------------------
    def get_tpr(self, pred, label, batch_dim=0):
        # return 1 under the division by zero scenario
        # return tpr per attributes
        numerator = torch.sum(self.tp_cells(pred, label), dim=batch_dim) # TP
        if self.soft_label:
            label = torch.where(label>0.4, 1, 0) # strong positive label
        denominator = torch.sum(label, dim=batch_dim) # all positive label
        tpr = torch.full_like(denominator, fill_value=1.0, dtype=torch.float)
        tpr_mask = (denominator != 0)
        tpr[tpr_mask] = numerator[tpr_mask]/denominator[tpr_mask]
        return tpr
    def get_tnr(self, pred, label, batch_dim=0):
        # return 1 under the division by zero scenario
        # return tnr per attributes
        numerator = torch.sum(self.tn_cells(pred, label), dim=batch_dim) # TN
        if self.soft_label:
            label = torch.where(label<0.6, 0, 1) # strong negative label
        denominator = torch.sum(torch.sub(1, label), dim=batch_dim) # all negative label
        tnr = torch.full_like(denominator, fill_value=1.0, dtype=torch.float)
        tnr_mask = (denominator != 0)
        tnr[tnr_mask] = numerator[tnr_mask]/denominator[tnr_mask]
        return tnr
    def get_fnr(self, pred, label, batch_dim=0):
        numerator = torch.sum(self.fn_cells(pred, label), dim=batch_dim) # FN
        if self.soft_label:
            label = torch.where(label>0.4, 1, 0) # strong positive label
        denominator = torch.sum(label, dim=batch_dim) # all positive label
        fnr = torch.full_like(denominator, fill_value=1.0, dtype=torch.float)
        fnr_mask = (denominator != 0)
        fnr[fnr_mask] = numerator[fnr_mask]/denominator[fnr_mask]
        return fnr
    def get_fpr(self, pred, label, batch_dim=0):
        numerator = torch.sum(self.fp_cells(pred, label), dim=batch_dim) # FP
        if self.soft_label:
            label = torch.where(label<0.6, 0, 1) # strong negative label
        denominator = torch.sum(torch.sub(1, label), dim=batch_dim) # all negative label
        fpr = torch.full_like(denominator, fill_value=1.0, dtype=torch.float)
        fpr_mask = (denominator != 0)
        fpr[fpr_mask] = numerator[fpr_mask]/denominator[fpr_mask]
        return fpr
    def get_acc(self, pred, label, batch_dim=0):
        numerator = torch.sum(self.tp_cells(pred, label), dim=batch_dim)+torch.sum(self.tn_cells(pred, label), dim=batch_dim) # TP+TN
        denominator = label.shape[batch_dim]
        accuracy = numerator/denominator
        return accuracy

    # -------------------- losses (binary) --------------------
    def get_bce_by_cells(self, logit, label, cells=['tp', 'fn', 'fp', 'tn']):
        # get the binary cross entropy loss per attributes with specific cm cells
        # get cells type
        pred = torch.where(logit> 0.5, 1, 0)
        cell_mask = torch.full_like(logit, fill_value=0.)
        for type in cells:
            match type:
                case 'tp':
                    cell_mask += self.tp_cells(pred, label)
                case 'fn':
                    cell_mask += self.fn_cells(pred, label)
                case 'fp':
                    cell_mask += self.fp_cells(pred, label)
                case 'tn':
                    cell_mask += self.tn_cells(pred, label)
                case _:
                    assert False, 'only "tp", "fn", "fp", "tn" are allowed'
        # mask with the bce loss
        bce_per_ins = F.binary_cross_entropy(logit, label, reduction='none')
        return torch.mul(bce_per_ins, cell_mask)

    def binary_direct_loss(self, logit, label, sens):
        # approximate prediction using a very steep function
        pred = 1./(1+torch.exp(-1e4*logit-0.5))
        group_1_pred, group_2_pred = regroup_tensor_binary(pred, sens, regroup_dim=0)
        group_1_label, group_2_label = regroup_tensor_binary(label, sens, regroup_dim=0)
        match self.fairness_criteria:
            case 'equality of opportunity':
                group_1_tpr = self.get_tpr(group_1_pred, group_1_label)
                group_2_tpr = self.get_tpr(group_2_pred, group_2_label)
                loss_per_attr = torch.abs(group_1_tpr-group_2_tpr)
            case 'equalized odds':
                group_1_tpr = self.get_tpr(group_1_pred, group_1_label)
                group_2_tpr = self.get_tpr(group_2_pred, group_2_label)
                group_1_tnr = self.get_tnr(group_1_pred, group_1_label)
                group_2_tnr = self.get_tnr(group_2_pred, group_2_label)
                loss_per_attr = torch.abs(group_1_tpr-group_2_tpr) + torch.abs(group_1_tnr-group_2_tnr)
            case _:
                assert False, f'unrecognized fairness criteria'
        return torch.mean(loss_per_attr)
    
    def binary_masking_loss(self, logit, label, sens):
        # find the relation between groups
        pred = 1./(1+torch.exp(-1e4*logit-0.5))
        group_1_pred, group_2_pred = regroup_tensor_binary(pred, sens, regroup_dim=0)
        group_1_label, group_2_label = regroup_tensor_binary(label, sens, regroup_dim=0)
        match self.fairness_criteria:
            case 'equality of opportunity':
                group_1_tpr = self.get_tpr(group_1_pred, group_1_label)
                group_2_tpr = self.get_tpr(group_2_pred, group_2_label)
                group_1_tpr_boost = torch.where(group_1_tpr < group_2_tpr, 1, 0)
                group_2_tpr_boost = 1-group_1_tpr_boost
                tp_bce = self.get_bce_by_cells(logit, label, ['tp',])
                group_1_tp_bce, group_2_tp_bce = regroup_tensor_binary(tp_bce, sens, regroup_dim=0)
                loss_per_attr = torch.sum(group_1_tp_bce*group_1_tpr_boost, dim=0)+\
                                torch.sum(group_2_tpr_boost*group_2_tp_bce, dim=0)
            case 'equalized odds':
                group_1_tpr = self.get_tpr(group_1_pred, group_1_label)
                group_2_tpr = self.get_tpr(group_2_pred, group_2_label)
                group_1_tnr = self.get_tnr(group_1_pred, group_1_label)
                group_2_tnr = self.get_tnr(group_2_pred, group_2_label)
                group_1_tpr_boost = torch.where(group_1_tpr < group_2_tpr, 1, 0)
                group_2_tpr_boost = 1-group_1_tpr_boost
                group_1_tnr_boost = torch.where(group_1_tnr < group_2_tnr, 1, 0)
                group_2_tnr_boost = 1-group_1_tnr_boost
                tp_bce = self.get_bce_by_cells(logit, label, ['tp',])
                tn_bce = self.get_bce_by_cells(logit, label, ['tn',])
                group_1_tp_bce, group_2_tp_bce = regroup_tensor_binary(tp_bce, sens, regroup_dim=0)
                group_1_tn_bce, group_2_tn_bce = regroup_tensor_binary(tn_bce, sens, regroup_dim=0)
                loss_per_attr = torch.sum(group_1_tp_bce*group_1_tpr_boost, dim=0)+ \
                                torch.sum(group_2_tpr_boost*group_2_tp_bce, dim=0)+ \
                                torch.sum(group_1_tn_bce*group_1_tnr_boost, dim=0)+ \
                                torch.sum(group_2_tnr_boost*group_2_tn_bce, dim=0)
            case _:
                assert False, f'unrecognized fairness criteria'
        return torch.mean(loss_per_attr)

    def binary_matching_loss(self, logit, label, sens):
        def split_tensor_by_label_attr(tensor, label, attr_idx, regroup_dim=0):
            match regroup_dim:
                case 0:
                    positive_tensor = tensor[label[:,attr_idx]==1]
                    negative_tensor = tensor[label[:,attr_idx]==0]
                case 1:
                    positive_tensor = tensor[:,label[:,attr_idx]==1]
                    negative_tensor = tensor[:,label[:,attr_idx]==0]
                case _:
                    assert False, 'regroup dimension only support 0 and 1'
            return positive_tensor[:,attr_idx], negative_tensor[:,attr_idx]
        # prediction and binary cross entropy
        pred = 1./(1+torch.exp(-1e4*logit-0.5))
        group_1_pred, group_2_pred = regroup_tensor_binary(pred, sens, regroup_dim=0)
        group_1_label, group_2_label = regroup_tensor_binary(label, sens, regroup_dim=0)
        bce = F.binary_cross_entropy(logit, label, reduction='none')
        group_1_bce, group_2_bce = regroup_tensor_binary(bce, sens, regroup_dim=0)
        match self.fairness_criteria:
            case 'equalized odds':
                # compute loss per attributes
                loss = 0
                for attr_idx in range(label.shape[1]):
                    group_1_bce_positive_label, group_1_bce_negative_label = split_tensor_by_label_attr(group_1_bce, group_1_label, attr_idx)
                    group_2_bce_positive_label, group_2_bce_negative_label = split_tensor_by_label_attr(group_2_bce, group_2_label, attr_idx)
                    loss += torch.abs(torch.mean(group_1_bce_positive_label)-torch.mean(group_2_bce_positive_label))
                    loss += torch.abs(torch.mean(group_1_bce_negative_label)-torch.mean(group_2_bce_negative_label))
        return loss

    def binary_perturb_loss(self, logit, label, sens):
        # fairness function
        def perturbed_eqopp(x, label=label, sens=sens):
            pred = torch.where(x> 0.5, 1, 0)
            label_duped = label.repeat(x.shape[0], 1, 1)
            group_1_pred, group_2_pred = regroup_tensor_binary(pred, sens, regroup_dim=1)
            group_1_label, group_2_label = regroup_tensor_binary(label_duped, sens, regroup_dim=1)
            # group_1_pred, group_2_pred = divide_tensor_binary(pred, divide_dim=1)
            # group_1_label, group_2_label = divide_tensor_binary(label_duped, divide_dim=1)
            group_1_tpr = self.get_tpr(group_1_pred, group_1_label, batch_dim=1)
            group_2_tpr = self.get_tpr(group_2_pred, group_2_label, batch_dim=1)
            return torch.abs(group_1_tpr-group_2_tpr)
        def perturbed_eqodd(x, label=label, sens=sens):
            pred = torch.where(x> 0.5, 1, 0)
            label_duped = label.repeat(x.shape[0], 1, 1)
            group_1_pred, group_2_pred = regroup_tensor_binary(pred, sens, regroup_dim=1)
            group_1_label, group_2_label = regroup_tensor_binary(label_duped, sens, regroup_dim=1)
            # group_1_pred, group_2_pred = divide_tensor_binary(pred, divide_dim=1)
            # group_1_label, group_2_label = divide_tensor_binary(label_duped, divide_dim=1)
            group_1_tpr = self.get_tpr(group_1_pred, group_1_label, batch_dim=1)
            group_2_tpr = self.get_tpr(group_2_pred, group_2_label, batch_dim=1)
            group_1_tnr = self.get_tnr(group_1_pred, group_1_label, batch_dim=1)
            group_2_tnr = self.get_tnr(group_2_pred, group_2_label, batch_dim=1)
            return torch.abs(group_1_tpr-group_2_tpr)+torch.abs(group_1_tnr-group_2_tnr)
        match self.fairness_criteria:
            case 'equality of opportunity':
                pret_eqopp = perturbed(perturbed_eqopp, 
                                       num_samples=10000,
                                       sigma=0.5,
                                       noise='gumbel',
                                       batched=False)
                loss_per_attr = pret_eqopp(logit)
            case 'equalized odds':
                pret_eqodd = perturbed(perturbed_eqodd, 
                                       num_samples=10000,
                                       sigma=0.5,
                                       noise='gumbel',
                                       batched=False)
                loss_per_attr = pret_eqodd(logit)
            case _:
                assert False, f'unrecognized fairness criteria'
        return torch.mean(loss_per_attr)

    def binary_accuracy_perturb_loss(self, logit, label, sens, p_coef, n_coef):
        def perturbed_FNR(x, label=label, sens=sens):
            pred = torch.where(x> 0.5, 1, 0)
            # dupe the label to have the same shape as x
            label_duped = label.repeat(x.shape[0], 1, 1)
            # regroup and compute FNR for both groups
            group_1_pred, group_2_pred = regroup_tensor_binary(pred, sens, regroup_dim=1)
            group_1_label, group_2_label = regroup_tensor_binary(label_duped, sens, regroup_dim=1)
            group_1_fnr = self.get_fnr(group_1_pred, group_1_label, batch_dim=1)
            group_2_fnr = self.get_fnr(group_2_pred, group_2_label, batch_dim=1)
            return group_1_fnr+group_2_fnr
        def perturbed_FPR(x, label=label, sens=sens):
            pred = torch.where(x> 0.5, 1, 0)
            # dupe the label to have the same shape as x
            label_duped = label.repeat(x.shape[0], 1, 1)
            # regroup and compute FNR for both groups
            group_1_pred, group_2_pred = regroup_tensor_binary(pred, sens, regroup_dim=1)
            group_1_label, group_2_label = regroup_tensor_binary(label_duped, sens, regroup_dim=1)
            group_1_fpr = self.get_fpr(group_1_pred, group_1_label, batch_dim=1)
            group_2_fpr = self.get_fpr(group_2_pred, group_2_label, batch_dim=1)
            return group_1_fpr+group_2_fpr
        
        pret_fnr = perturbed(perturbed_FNR,
                             num_samples=10000,
                             sigma=0.5,
                             noise='gumbel',
                             batched=False)
        pret_fpr = perturbed(perturbed_FPR,
                             num_samples=10000,
                             sigma=0.5,
                             noise='gumbel',
                             batched=False)

        match self.fairness_criteria:
            case 'equality of opportunity':
                loss_per_attr = pret_fnr(logit)*p_coef
            case 'equalized odds':
                loss_per_attr = pret_fnr(logit)*p_coef + pret_fpr(logit)*n_coef
            case _:
                assert False, f'unrecognized fairness criteria'
        return torch.mean(loss_per_attr)

    # -------------------- losses (categorical) --------------------
    def categori_filter_logit(self, logit, batch_dim=0):
        match (self.dataset_name, batch_dim):
            case ("UTKFace", 0):
                return logit[:,0:2], logit[:,2:11]
            case ("UTKFace", 1):
                return logit[:,:,0:2], logit[:,:,2:11]
            case ("HAM10000", 0) | ("FairFace", 0):
                return logit[:,0:7]
            case ("HAM10000", 1) | ("FairFace", 1):
                return logit[:,:,0:7]
            case _:
                assert False, "batch dimension must be 0 or 1"
    
    def categori_to_prediction(self, logit, batch_dim=0):
        # encode in index
        _, pred = torch.max(logit, dim=batch_dim+1)
        return pred
    
    def categori_to_approx_prediction(self, logit, batch_dim=0):
        # encode in one hot 
        softmaxed = F.softmax(logit, dim=batch_dim+1)
        pred = 1./(1+torch.exp(-1e2*(softmaxed-0.5)))
        return pred

    def categori_direct_loss(self, logit, label, sens):
        match self.dataset_name:
            case "UTKFace":
                # Gender, Age
                gender_logit, age_logit = self.categori_filter_logit(logit, batch_dim=0)
                age_pred = self.categori_to_approx_prediction(logit=age_logit, batch_dim=0)
                age_label = F.one_hot(label[:,1], num_classes=9)
                result = torch.sum(age_pred*age_label, dim=1) # with shape (N)
            case "HAM10000" | "FairFace":
                # Case (Diagnosis)
                case_logit = self.categori_filter_logit(logit, batch_dim=0)
                case_pred = self.categori_to_approx_prediction(logit=case_logit, batch_dim=0)
                case_label = F.one_hot(label[:,0], num_classes=7)
                result = torch.sum(case_pred*case_label, dim=1) # with shape (N)
        group_1_result, group_2_result = regroup_tensor_categori(result, sens, regroup_dim=0)
        group_1_total, group_2_total = group_1_result.shape[0], group_2_result.shape[0]
        group_1_acc, group_2_acc = group_1_result.sum()/(group_1_total+1e-9), group_2_result.sum()/(group_2_total+1e-9)
        loss = torch.abs(group_1_acc-group_2_acc) # difference in accuracy
        return loss # a single value

    def categori_masking_loss(self, logit, label, sens):
        match self.dataset_name:
            case "UTKFace":
                # Gender, Age
                gender_logit, age_logit = self.categori_filter_logit(logit, batch_dim=0)
                age_pred = self.categori_to_approx_prediction(logit=age_logit, batch_dim=0)
                age_label = F.one_hot(label[:,1], num_classes=9)
                result = torch.sum(age_pred*age_label, dim=1) # with shape (N)
                loss_CE = F.cross_entropy(age_logit, label[:,1], reduction='none') # with shape (N)
            case "HAM10000" | "FairFace":
                # Case (Diagnosis)
                case_logit = self.categori_filter_logit(logit, batch_dim=0)
                case_pred = self.categori_to_approx_prediction(logit=case_logit, batch_dim=0)
                case_label = F.one_hot(label[:,0], num_classes=7)
                result = torch.sum(case_pred*case_label, dim=1) # with shape (N)
                loss_CE = F.cross_entropy(case_logit, label[:,0], reduction='none') # with shape (N)
        group_1_result, group_2_result = regroup_tensor_categori(result, sens, regroup_dim=0)
        group_1_total, group_2_total = group_1_result.shape[0], group_2_result.shape[0]
        group_1_acc, group_2_acc = group_1_result.sum()/(group_1_total+1e-9), group_2_result.sum()/(group_2_total+1e-9)
        group_1_coef, group_2_coef = torch.where(group_1_acc < group_2_acc, 1, 0), torch.where(group_2_acc < group_1_acc, 1, 0)
        group_1_CE, group_2_CE = regroup_tensor_categori(loss_CE, sens, regroup_dim=0)
        loss = torch.cat((group_1_CE*group_1_coef, group_2_CE*group_2_coef), 0) # with shape (N)
        return torch.mean(loss)
    
    def categori_matching_loss(self, logit, label, sens):
        match self.dataset_name:
            case "UTKFace":
                # Gender, Age
                gender_logit, age_logit = self.categori_filter_logit(logit, batch_dim=0)
                age_pred = self.categori_to_approx_prediction(logit=age_logit, batch_dim=0)
                age_label = F.one_hot(label[:,1], num_classes=9)
                result = torch.sum(age_pred*age_label, dim=1) # with shape (N)
                loss_CE = F.cross_entropy(age_logit, label[:,1], reduction='none') # with shape (N)
            case "HAM10000" | "FairFace":
                # Case (Diagnosis)
                case_logit = self.categori_filter_logit(logit, batch_dim=0)
                case_pred = self.categori_to_approx_prediction(logit=case_logit, batch_dim=0)
                case_label = F.one_hot(label[:,0], num_classes=7)
                result = torch.sum(case_pred*case_label, dim=1) # with shape (N)
                loss_CE = F.cross_entropy(case_logit, label[:,0], reduction='none') # with shape (N)
        group_1_CE, group_2_CE = regroup_tensor_categori(loss_CE, sens, regroup_dim=0)
        loss = torch.abs(torch.mean(group_1_CE)-torch.mean(group_2_CE))
        return loss

    def categori_perturb_loss(self, logit, label, sens):
        def perturbed_pq(x, label=label):
            match self.dataset_name:
                case "UTKFace":
                    # Gender, Age
                    gender_logit, age_logit = self.categori_filter_logit(x, batch_dim=1) # logit in shape (P, N)
                    age_pred = self.categori_to_prediction(logit=age_logit, batch_dim=1)
                    age_label_duped = label[:,1].repeat(x.shape[0], 1) # to shape (P, N)
                    result = torch.eq(age_pred, age_label_duped) # with shape (P, N)
                case "HAM10000" | "FairFace":
                    # Case (Diagnosis)
                    case_logit = self.categori_filter_logit(x, batch_dim=1) # logit in shape (P, N)
                    case_pred = self.categori_to_prediction(logit=case_logit, batch_dim=1)
                    case_label_duped = label[:,0].repeat(x.shape[0], 1) # to shape (P, N)
                    result = torch.eq(case_pred, case_label_duped) # with shape (P, N)
            group_1_result, group_2_result = regroup_tensor_categori(result, sens, regroup_dim=1)
            group_1_total, group_2_total = group_1_result.shape[1], group_2_result.shape[1]
            group_1_acc, group_2_acc = torch.sum(group_1_result, dim=1)/(group_1_total+1e-9), torch.sum(group_2_result, dim=1)/(group_2_total+1e-9)
            perturbed_loss = torch.abs(group_1_acc-group_2_acc)
            return perturbed_loss
        # Turns a function into a differentiable one via perturbations
        pret_pq = perturbed(perturbed_pq, 
                             num_samples=10000,
                             sigma=0.5,
                             noise='gumbel',
                             batched=False)
        # loss perturbed_loss
        loss = pret_pq(logit) # a single value
        return loss

    def categori_accuracy_perturb_loss(self, logit, label, sens, coef):
        def perturb_acc(x, label=label):
            match self.dataset_name:
                case "UTKFace":
                    # Gender, Age
                    gender_logit, age_logit = self.categori_filter_logit(x, batch_dim=1) # logit in shape (P, N)
                    age_pred = self.categori_to_prediction(logit=age_logit, batch_dim=1)
                    age_label_duped = label[:,1].repeat(x.shape[0], 1) # to shape (P, N)
                    result = torch.eq(age_pred, age_label_duped) # with shape (P, N)
                case "HAM10000" | "FairFace":
                    # Case (Diagnosis)
                    case_logit = self.categori_filter_logit(x, batch_dim=1) # logit in shape (P, N)
                    case_pred = self.categori_to_prediction(logit=case_logit, batch_dim=1)
                    case_label_duped = label[:,0].repeat(x.shape[0], 1) # to shape (P, N)
                    result = torch.eq(case_pred, case_label_duped) # with shape (P, N)
            group_1_result, group_2_result = regroup_tensor_categori(result, sens, regroup_dim=1)
            group_1_total, group_2_total = group_1_result.shape[1], group_2_result.shape[1]
            group_1_acc, group_2_acc = torch.sum(group_1_result, dim=1)/(group_1_total+1e-9), torch.sum(group_2_result, dim=1)/(group_2_total+1e-9)
            perturbed_loss = ((1 - group_1_acc) + (1 - group_2_acc))
            return perturbed_loss
        # Turns a function into a differentiable one via perturbations
        pret_acc = perturbed(perturb_acc, 
                             num_samples=10000,
                             sigma=0.5,
                             noise='gumbel',
                             batched=False)
        loss = pret_acc(logit)*coef
        return loss

    def run(self, logit, label, sens, coef, n_coef=None):
        recovery_loss = 0
        if self.pred_type == 'binary':
            if len(coef) and sum(coef) > 0:
                match self.loss_type:
                    case 'direct' | 'masking' | 'matching' | 'perturb optim':
                        pred = torch.where(logit> 0.5, 1, 0)
                        FN_mask, FP_mask = torch.sub(1, pred)*label, pred*torch.sub(1, label)
                        BCEloss = F.binary_cross_entropy(logit, label, reduction='none')
                        FN_BCE, FP_BCE = torch.mean(BCEloss*FN_mask, dim=0), torch.mean(BCEloss*FP_mask, dim=0)
                        recovery_loss = torch.mean(coef*FN_BCE+n_coef*FP_BCE)
                    case 'full perturb optim' | 'masking perturb recovery':
                        recovery_loss = self.binary_accuracy_perturb_loss(logit, label, sens, coef, n_coef)
            match self.loss_type:
                case 'direct':
                    loss = self.binary_direct_loss(logit, label, sens)
                case 'masking' | 'masking perturb recovery':
                    loss = self.binary_masking_loss(logit, label, sens)
                case 'matching':
                    loss = self.binary_matching_loss(logit, label, sens)
                case 'perturb optim' | 'full perturb optim':
                    loss = self.binary_perturb_loss(logit, label, sens)
                case _:
                    assert False, f'do not support such loss type'
        elif self.pred_type == 'categorical':
            if len(coef) and sum(coef) > 0:
                match self.loss_type:
                    case 'direct' | 'masking' | 'matching' | 'perturb optim':
                        match self.dataset_name:
                            case "UTKFace":
                                _, age_logit = self.categori_filter_logit(logit, batch_dim=0)
                                loss_CE = F.cross_entropy(age_logit, label[:,1], reduction='none') # with shape (N)
                            case "HAM10000" | "FairFace":
                                case_logit = self.categori_filter_logit(logit, batch_dim=0)
                                loss_CE = F.cross_entropy(case_logit, label[:,0], reduction='none') # with shape (N)
                        recovery_loss = torch.mean(loss_CE, dim=0) * coef
                    case 'full perturb optim' | 'masking perturb recovery':
                        recovery_loss = self.categori_accuracy_perturb_loss(logit, label, sens, coef)
            match self.loss_type:
                case 'direct':
                    loss = self.categori_direct_loss(logit, label, sens)
                case 'masking' | 'masking perturb recovery':
                    loss = self.categori_masking_loss(logit, label, sens)
                case 'matching':
                    loss = self.categori_matching_loss(logit, label, sens)
                case 'perturb optim' | 'full perturb optim':
                    loss = self.categori_perturb_loss(logit, label, sens)
                case _:
                    assert False, f'do not support such loss type'
        return loss + recovery_loss