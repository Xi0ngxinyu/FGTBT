# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------

import math
import random

import torch
import torch.utils.data as data
import numpy as np

from hdf5storage import loadmat
from .transforms import fliplr_joints, crop, generate_target, transform_pixel, draw_gaussian
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class COFW(data.Dataset):

    def __init__(self, cfg, is_train=True, transform=None):
        # specify annotation file for dataset
        if is_train:
            self.mat_file = cfg.DATASET_COFW.TRAINSET
        else:
            self.mat_file = cfg.DATASET_COFW.TESTSET

        self.is_train = is_train
        self.transform = transform
        self.input_size = cfg.MODEL.IMAGE_SIZE
        self.output_size = cfg.MODEL.HEATMAP_SIZE
        self.scale_factor = cfg.DATASET_COFW.SCALE_FACTOR
        self.rot_factor = cfg.DATASET_COFW.ROT_FACTOR
        self.flip = cfg.DATASET_COFW.FLIP

        # load annotations
        self.mat = loadmat(self.mat_file)
        if is_train:
            images = self.mat['IsTr']
            pts = self.mat['phisTr']
            data_size = len(images)
            repeat_num = int(cfg.TRAIN.LARGEST_NUM / data_size) + 1
            dataset_dicts_new = []
            dataset_pts_new = []
            for ii in range(repeat_num):
                dataset_dicts_new = dataset_dicts_new + list(images)
                dataset_pts_new = dataset_pts_new + list(pts)
            self.images = dataset_dicts_new[:cfg.TRAIN.LARGEST_NUM]
            self.pts = dataset_pts_new[:cfg.TRAIN.LARGEST_NUM]
        else:
            self.images = self.mat['IsT']
            self.pts = self.mat['phisT']

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img = self.images[idx][0]

        if len(img.shape) == 2:
            img = img.reshape(img.shape[0], img.shape[1], 1)
            img = np.repeat(img, 3, axis=2)

        pts = self.pts[idx][0:58].reshape(2, -1).transpose()

        # show_preds(img, pts)

        xmin = np.min(pts[:, 0])
        xmax = np.max(pts[:, 0])
        ymin = np.min(pts[:, 1])
        ymax = np.max(pts[:, 1])

        center_w = (math.floor(xmin) + math.ceil(xmax)) / 2.0
        center_h = (math.floor(ymin) + math.ceil(ymax)) / 2.0

        scale = max(math.ceil(xmax) - math.floor(xmin), math.ceil(ymax) - math.floor(ymin)) / 200.0
        center = torch.Tensor([center_w, center_h])

        scale *= 1.25
        nparts = pts.shape[0]

        r = 0
        if self.is_train:
            scale = scale * (random.uniform(1 - self.scale_factor,
                                            1 + self.scale_factor))
            r = random.uniform(-self.rot_factor, self.rot_factor) \
                if random.random() <= 0.6 else 0

            if random.random() <= 0.5 and self.flip:
                img = np.fliplr(img)
                pts = fliplr_joints(pts, width=img.shape[1], dataset='COFW')
                center[0] = img.shape[1] - center[0]

        img = crop(img, center, scale, self.input_size, rot=r)

        # target = np.zeros((nparts, self.output_size[0], self.output_size[1]))
        tpts = pts.copy()

        for i in range(nparts):
            if tpts[i, 1] > 0:
                tpts[i, 0:2] = transform_pixel(tpts[i, 0:2]+1, center, scale, self.output_size, rot=r)
                # target[i] = generate_target(target[i], tpts[i]-1, self.sigma,
                #                             label_type=self.label_type)
        # show_preds(img, tpts)

        img = img.astype(np.float32)
        img = (img/255 - self.mean) / self.std
        img = img.transpose([2, 0, 1])
        target = tpts[:, 0:2]
        tpts = torch.Tensor(tpts)
        center = torch.Tensor(center)
        target = torch.Tensor(target)

        num_points = target.shape[0]
        # target = target.astype(np.int32)
        target = target.int()
        heatmap = np.zeros([num_points, self.output_size[0], self.output_size[1]])
        for n in range(num_points):
            heatmap[n] = draw_gaussian(heatmap[n], target[n], sigma=3)
        target_heatmap = torch.from_numpy(heatmap).float()   # 真实值对应的热图

        meta = {'index': idx, 'center': center, 'scale': scale, 'img_pth': idx,
                'rotate': r, 'pts': torch.Tensor(pts), 'tpts': tpts,'output_size': self.output_size}

        return img, target, target_heatmap, meta


if __name__ == '__main__':

    pass
