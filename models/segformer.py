import torch
import torch.nn as nn

from models import register
from losses import awingLoss, anisotropicDirectionLoss, smoothL1Loss
from .mmseg.models import build_segmentor
from mmseg.models.builder import BACKBONES, SEGMENTORS
import logging

logger = logging.getLogger(__name__)
# cofw1=[]
# cofw2=[5, 7, 12, 13, 14, 15, 18, 19]
# cofw3=[16, 17, 21, 24, 25, 27]
# cofw4=[0, 1, 2, 3, 4, 6, 8, 9, 10, 11, 20, 22, 23, 26, 28]
# aflw1=[]
# aflw2=[12, 14]
# aflw3=[7, 10]
# aflw4=[0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 13, 15, 16, 17, 18]
# _300w1=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 37, 38, 40, 41, 43, 44, 46, 47]
# _300w2=[18, 20, 23, 25, 27, 28, 29, 31, 32, 34, 35, 49, 50, 52, 53, 55, 56, 58, 59, 60, 61, 63, 64, 65, 67]
# _300w3=[33, 51, 57, 62]
# _300w4=[8, 17, 19, 21, 22, 24, 26, 30, 36, 39, 42, 45, 48, 54, 66]
# wflw1=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 38, 39, 41, 47, 49, 50, 61, 63, 65, 67, 69, 71, 73, 75]
# wflw2=[34, 36, 40, 43, 45, 48, 51, 52, 53, 55, 56, 58, 59, 62, 66, 70, 74, 77, 78, 80, 81, 83, 84, 86, 87, 88, 89, 91, 92, 93, 95]
# wflw3=[57, 79, 85, 90, 96, 97]
# wflw4=[16, 33, 35, 37, 42, 44, 46, 54, 60, 64, 68, 72, 76, 82, 94]
# part_map = {
#     0: torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27,
#                      28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 54, 55, 57, 63,
#                      65, 66, 79, 80, 82, 83, 85, 86, 88, 89, 91, 92, 94, 95, 97, 98, 100, 101], device="cuda"),
#     1: torch.tensor([50, 52, 56, 59, 61, 64, 67, 68, 69, 71, 72, 74, 75, 76, 77, 81, 87, 93, 99, 103, 104, 106, 107,
#                      109, 110, 112, 113, 114, 115, 117, 118, 119, 121], device="cuda"),
#     2: torch.tensor([73, 105, 111, 116, 122, 123], device="cuda"),
#     3: torch.tensor([24, 49, 51, 53, 58, 60, 62, 70, 78, 84, 90, 96, 102, 108, 120], device="cuda")
# }

# parts_map = {
#     29: [cofw1, cofw2, cofw3, cofw4],
#     68: [_300w1, _300w2, _300w3, _300w4],
#     19: [aflw1, aflw2, aflw3, aflw4],
#     98: [wflw1, wflw2, wflw3, wflw4]
# }
# fre =torch.tensor([70,66,18,60]).to("cuda")


# def init_weights(layer):
#     if type(layer) == nn.Conv2d:
#         nn.init.normal_(layer.weight, mean=0.0, std=0.02)
#         nn.init.constant_(layer.bias, 0.0)
#     elif type(layer) == nn.Linear:
#         nn.init.normal_(layer.weight, mean=0.0, std=0.02)
#         nn.init.constant_(layer.bias, 0.0)
#     elif type(layer) == nn.BatchNorm2d:
#         # print(layer)
#         nn.init.normal_(layer.weight, mean=1.0, std=0.02)
#         nn.init.constant_(layer.bias, 0.0)


class BBCEWithLogitLoss(nn.Module):
    '''
    Balanced BCEWithLogitLoss
    '''
    def __init__(self):
        super(BBCEWithLogitLoss, self).__init__()

    def forward(self, pred, gt):  # pred模型输出结果 groundtruth
        eps = 1e-10
        count_pos = torch.sum(gt) + eps
        count_neg = torch.sum(1. - gt)
        ratio = count_neg / count_pos
        w_neg = count_pos / (count_pos + count_neg)  # weight_neg

        bce1 = nn.BCEWithLogitsLoss(pos_weight=ratio)  # 二分类的带权重的交叉熵损失函数
        loss = w_neg * bce1(pred, gt)

        return loss


def _iou_loss(pred, target):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3)) - inter
    iou = 1 - (inter / union)

    return iou.mean()


@register('segformer')
class SegFormer(nn.Module):
    def __init__(self, train_batch_size=None, num_points=None, extra=None, inp_size=None, encoder_mode=None, loss=None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 选择特定的encoder类型 这里是evp
        if encoder_mode['name'] == 'vpt_deep':
            backbone = dict(
                type=BACKBONES.get('mit_b4_vpt'),
                img_size=inp_size,
                prompt_cfg=encoder_mode['name'],)
        elif encoder_mode['name'] == 'fp':
            backbone = dict(
                type=BACKBONES.get('mit_b4_fp'),
                img_size=inp_size,
                scale_factor=encoder_mode['scale_factor'],
                tuning_stage=encoder_mode['tuning_stage'],
                frequency_tune=encoder_mode['frequency_tune'],
                embedding_tune=encoder_mode['embedding_tune'],
                adaptor=encoder_mode['adaptor'])
        elif encoder_mode['name'] == 'evp':
            backbone = dict(
                type=BACKBONES.get('mit_b4_evp'),
                img_size=inp_size,
                scale_factor=encoder_mode['scale_factor'],
                prompt_type=encoder_mode['prompt_type'],
                tuning_stage=encoder_mode['tuning_stage'],
                input_type=encoder_mode['input_type'],
                freq_nums=encoder_mode['freq_nums'],
                handcrafted_tune=encoder_mode['handcrafted_tune'],
                embedding_tune=encoder_mode['embedding_tune'],
                adaptor=encoder_mode['adaptor'],
                batch_size=encoder_mode['batch_size']
            )
        elif encoder_mode['name'] == 'linear':
            backbone = dict(
                type=BACKBONES.get('mit_b4'),
                img_size=inp_size)
        elif encoder_mode['name'] == 'adaptformer':
            backbone = dict(
                type=BACKBONES.get('mit_b4_adaptformer'),
                img_size=inp_size)
        else:
            backbone = dict(
                type=BACKBONES.get('mit_b4'),
                img_size=inp_size)

        model_config = dict(
            type='EncoderDecoder',
            pretrained='../mit_b4.pth',
            backbone=backbone,
            decode_head=dict(
                type='SegFormerHead',
                in_channels=[64, 128, 320, 512],  # [64, 128, 320, 512]
                in_index=[0, 1, 2, 3],
                feature_strides=[4, 8, 16, 32],
                channels=128,
                dropout_ratio=0.1,
                num_classes=num_points,  # 1
                batch_size=train_batch_size,  # 后来加的
                image_size=inp_size,  # 后来加的
                norm_cfg=dict(type='BN', requires_grad=True),
                align_corners=False,
                decoder_params=dict(embed_dim=768),  # 原来768
                loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
            # model training and testing settings
            train_cfg=dict(),
            test_cfg=dict(mode='whole'))

        # print('loading segformer weigths...')
        model = build_segmentor(
            model_config,
            # train_cfg=dict(),
            # test_cfg=dict(mode='whole')
        )  # 这里加载了模型

        self.encoder = model

        # if encoder_mode['name'] == 'evp':
        #     for k, p in self.encoder.named_parameters():
        #         if "prompt" not in k and "decode_head" not in k:
        #             p.requires_grad = False
        # if encoder_mode['name'] == 'vpt_deep':
        #     for k, p in self.encoder.named_parameters():
        #         if "prompt" not in k and "decode_head" not in k:
        #             p.requires_grad = False
        # if encoder_mode['name'] == 'adaptformer':
        #     for k, p in self.encoder.named_parameters():
        #         if "adaptmlp" not in k and "decode_head" not in k:
        #             p.requires_grad = False
        # if encoder_mode['name'] == 'linear':
        #     for k, p in self.encoder.named_parameters():
        #         if "decode_head" not in k:
        #             p.requires_grad = False

        # model_total_params = sum(p.numel() for p in self.encoder.parameters())
        # model_grad_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        # print('model_grad_params:' + str(model_grad_params),
        #       '\nmodel_total_params:' + str(model_total_params))

        self.loss_mode = loss
        if self.loss_mode == 'MSELoss':
            self.criterion = torch.nn.MSELoss()

        elif self.loss_mode == "AWingLoss":
            self.criterion = awingLoss.AWingLoss()

        elif self.loss_mode == "AnisotropicDirectionLoss":
            self.criterion = anisotropicDirectionLoss.AnisotropicDirectionLoss()
        # (loss_lambda=config.loss_lambda,edge_info=config.edge_info)
        elif self.loss_mode == "SmoothL1Loss":
            self.criterion = smoothL1Loss.SmoothL1Loss()

        # params = torch.ones(1, requires_grad=True)
        # self.params = torch.nn.Parameter(params)
        # print("")

    def set_input(self, input, gt_mask, heatmap, batch_size):
        for i in range(4):
            gt_mask[i] = gt_mask[i].to(self.device)
            heatmap[i] = heatmap[i].to(self.device)
            input[i] = input[i].to(self.device)
        self.input = torch.cat(input, dim=0)
        self.gt_mask = gt_mask
        self.heatmap = heatmap
        self.batch_size = batch_size

    def forward(self):
        self.pred_mask = self.encoder.forward_dummy(self.input)

    def set_landmark_index(self, _cofw_landmark_index, _aflw_landmark_index, _300w_landmark_index,
                           _wflw_landmark_index, weight, beta):
        self._cofw_landmark_index = _cofw_landmark_index
        self._aflw_landmark_index = _aflw_landmark_index
        self._300w_landmark_index = _300w_landmark_index
        self._wflw_landmark_index = _wflw_landmark_index
        weight = torch.tensor(weight, device="cuda")
        self.weight124 = (1-beta)/(1-beta**weight).to("cuda")

    def backward_G(self):
        """Calculate loss"""
        # losses = torch.zeros(124, device="cuda")
        losses = []
        for i in range(len(self.heatmap)):
            num_joints = self.heatmap[i].shape[1]
            if num_joints == 29:
                landmark_index = self._cofw_landmark_index
            elif num_joints == 68:
                landmark_index = self._300w_landmark_index
            elif num_joints == 19:
                landmark_index = self._aflw_landmark_index
            else:
                landmark_index = self._wflw_landmark_index
            pred_mask = self.pred_mask[i*self.batch_size:(i+1)*self.batch_size, landmark_index, :, :]
            weight = self.weight124[landmark_index]
            loss = self.criterion(pred_mask, self.heatmap[i], weight)
            losses.append(loss)
            # losses[landmark_index] += loss
        # losses *= self.weight124
        self.loss_G = sum(losses)
        self.loss_G.backward()

    def optimize_parameters(self):
        # self.epoch = epoch-1
        self.forward()
        self.optimizer.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer.step()  # udpate G's weights

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
