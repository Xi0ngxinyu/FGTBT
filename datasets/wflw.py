# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------

import os
import random

import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np

from .transforms import fliplr_joints, crop, generate_target, transform_pixel_float, draw_gaussian


class WFLW(data.Dataset):
    def __init__(self, cfg, is_train=True, transform=None):
        # specify annotation file for dataset
        if is_train:
            self.csv_file = cfg.DATASET_WFLW.TRAINSET
            landmarks_frame = pd.read_csv(self.csv_file)
            extended_df = pd.concat([landmarks_frame] * (cfg.TRAIN.LARGEST_NUM // len(landmarks_frame) + 1),
                                    ignore_index=True)
            self.landmarks_frame = extended_df.iloc[:cfg.TRAIN.LARGEST_NUM]
        else:
            self.csv_file = cfg.DATASET_WFLW.TESTSET
            self.landmarks_frame = pd.read_csv(self.csv_file)

        self.is_train = is_train
        self.transform = transform
        self.data_root = cfg.DATASET_WFLW.ROOT
        self.input_size = cfg.MODEL.IMAGE_SIZE
        self.output_size = cfg.MODEL.HEATMAP_SIZE
        self.scale_factor = cfg.DATASET_WFLW.SCALE_FACTOR
        self.rot_factor = cfg.DATASET_WFLW.ROT_FACTOR
        self.flip = cfg.DATASET_WFLW.FLIP

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):

        image_path = os.path.join(self.data_root,
                                  self.landmarks_frame.iloc[idx, 0])
        scale = self.landmarks_frame.iloc[idx, 1]

        center_w = self.landmarks_frame.iloc[idx, 2]
        center_h = self.landmarks_frame.iloc[idx, 3]
        center = torch.Tensor([center_w, center_h])

        pts = self.landmarks_frame.iloc[idx, 4:].values
        pts = pts.astype('float').reshape(-1, 2)

        scale *= 1.25
        nparts = pts.shape[0]
        img = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)

        r = 0
        if self.is_train:
            scale = scale * (random.uniform(1 - self.scale_factor,
                                            1 + self.scale_factor))
            r = random.uniform(-self.rot_factor, self.rot_factor) \
                if random.random() <= 0.6 else 0
            if random.random() <= 0.5 and self.flip:
                img = np.fliplr(img)
                pts = fliplr_joints(pts, width=img.shape[1], dataset='WFLW')
                center[0] = img.shape[1] - center[0]

        img = crop(img, center, scale, self.input_size, rot=r)

        # target = np.zeros((nparts, self.output_size[0], self.output_size[1]))
        tpts = pts.copy()

        # # 小数精度范围256的变换后的坐标
        for i in range(nparts):
            if tpts[i,1] >0:
                tpts[i, 0:2] = transform_pixel_float(tpts[i, 0:2]+1, center,
                                               scale, self.output_size, rot=r)

        img = img.astype(np.float32)
        img = (img/255.0 - self.mean) / self.std
        img = img.transpose([2, 0, 1])
        target = tpts[:, 0:2]
        tpts = torch.Tensor(tpts)
        center = torch.Tensor(center)
        target = torch.tensor(target)

        num_points = target.shape[0]
        # target = target.astype(np.int32)
        target = target.int()
        heatmap = np.zeros([num_points, self.output_size[0], self.output_size[1]])
        for n in range(num_points):
            heatmap[n] = draw_gaussian(heatmap[n], target[n], sigma=3)
        target_heatmap = torch.from_numpy(heatmap).float()

        meta = {'index': idx, 'center': center, 'scale': scale, 'img_pth': image_path,
                'rotate': r, 'pts': torch.Tensor(pts), 'tpts': tpts,'output_size': self.output_size}

        return img, target, target_heatmap, meta


if __name__ == '__main__':
    pass
