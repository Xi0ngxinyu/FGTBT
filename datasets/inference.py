# 开发时间：2024/02/29 15:27
# ------------------------------------------------------------------------------
# Modified from HRNet-Human-Pose-Estimation
# (https://github.com/HRNet/HRNet-Human-Pose-Estimation)
# Copyright (c) Microsoft
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torch.nn.functional as F
import torch
from scipy.optimize import linear_sum_assignment
from datasets.transforms import get_transform, transform_preds

match_parts_68 = np.array([[0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11], [6, 10], [7, 9],  # outline
                           [17, 26], [18, 25], [19, 24], [20, 23], [21, 22],  # eyebrow
                           [36, 45], [37, 44], [38, 43], [39, 42], [40, 47], [41, 46],  # eye
                           [31, 35], [32, 34],  # nose
                           [48, 54], [49, 53], [50, 52], [59, 55], [58, 56],  # outer mouth
                           [60, 64], [61, 63], [67, 65]])

match_parts_98 = np.array(
    [[0, 32], [1, 31], [2, 30], [3, 29], [4, 28], [5, 27], [6, 26], [7, 25], [8, 24], [9, 23], [10, 22], [11, 21],
     [12, 20], [13, 19], [14, 18], [15, 17],  # outline
     [33, 46], [34, 45], [35, 44], [36, 43], [37, 42], [41, 47], [40, 48], [39, 49], [38, 50],  # eyebrow
     [60, 72], [61, 71], [62, 70], [63, 69], [64, 68], [67, 73], [66, 74], [65, 75], [96, 97],  # eye
     [55, 59], [56, 58],  # nose
     [76, 82], [77, 81], [78, 80], [87, 83], [86, 84],  # outer mouth
     [88, 92], [89, 91], [95, 93]])


def flippoints(kps, width):
    nPoints = kps.shape[0]
    assert nPoints in (68, 98), 'flip {} nPoints is not supported'
    if nPoints == 98:
        pairs = match_parts_98
    else:
        pairs = match_parts_68
    fkps = kps.copy()

    for pair in pairs:
        fkps[pair[0]] = kps[pair[1]]
        fkps[pair[1]] = kps[pair[0]]
    fkps[:, 0] = width - fkps[:, 0] - 1

    return fkps


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_final_preds(config, batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                    diff = np.array(
                        [
                            hm[py][px + 1] - hm[py][px - 1],
                            hm[py + 1][px] - hm[py - 1][px]
                        ]
                    )
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )

    return preds, maxvals


def get_coords(preds, center, scale, rotate, output_size):
    c, _ = preds.shape  ## 对应一幅图的点
    coords = torch.zeros((c, 2))
    t = get_transform(center, scale, output_size, rotate)
    t = np.linalg.inv(t)
    for j in range(c):
        pt = preds[j, :]
        new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
        new_pt = np.dot(t, new_pt)
        coords[j, :] = torch.from_numpy(new_pt[:2])
    return coords


def get_final_preds_match(config, outputs, center, scale, rotate, flip=False):
    pred_logits = outputs['pred_logits'].detach().cpu()
    pred_coords = outputs['pred_coords'].detach().cpu()
    num_joints = pred_logits.shape[-1] - 1

    # if config.TEST.INCLUDE_BG_LOGIT:
    #     prob = F.softmax(pred_logits, dim=-1)[..., :-1]
    # else:
    prob = F.softmax(pred_logits[..., :-1], dim=-1)

    score_holder = []
    coord_holder = []
    orig_coord = []
    for b, C in enumerate(prob):
        _, query_ind = linear_sum_assignment(-C.transpose(0, 1))  # Cost Matrix: [17, N]
        score = prob[b, query_ind, list(np.arange(num_joints))][..., None].numpy()
        pred_raw = pred_coords[b, query_ind].numpy()
        # scale to the whole patch
        pred_raw *= np.array(config.MODEL.IMAGE_SIZE)

        if flip:
            pred_raw = flippoints(pred_raw, config.MODEL.IMAGE_SIZE[0])

        # transform back w.r.t. the entire img
        pred = get_coords(pred_raw, center[b], scale[b], rotate[b], config.MODEL.IMAGE_SIZE)
        orig_coord.append(pred_raw)
        score_holder.append(score)
        coord_holder.append(pred)

    matched_score = np.stack(score_holder)
    matched_coord = np.stack(coord_holder)

    return matched_coord, matched_score, np.stack(orig_coord)

