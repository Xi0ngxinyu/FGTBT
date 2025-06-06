# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict
import torch.nn.functional as F
from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from ..backbones.mix_transformer_evp import Block
from mmseg.models.utils import *
import attr

from IPython import embed


#就是用来创建一个3*3的卷积核
def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    #bias解释：偏置项在卷积神经网络中引入了非线性偏移，允许神经元在没有输入信号时仍能产生输出，并增加了网络的灵活性和表示能力。
    "3x3 convolution with padding"
    #conv2d用来创建二维卷积核，后面参数为输入通道数，输出通道数，卷积核大小（边长），步长，填充大小，是否包含偏置项
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=1, padding=1, bias=False)

#实现了一个卷积块的功能
class ConvBlock(nn.Module):#在类后面写一个参数表示继承另外一个类
    def __init__(self, in_planes, out_planes):#输入通道数，输出通道数
        #下面这句表示执行了ConvBlock的父类（nn.Module）的构造函数
        super(ConvBlock, self).__init__()
        planes = int(out_planes / 2)
        #相当于写了bn1这个函数，功能和BatchNorm2d一样
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        #用3*3的卷积核卷积
        self.conv2 = conv3x3(planes, planes)

        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, bias=False)

        # 将输入通道数和输出通道数改为一致
        if in_planes != out_planes:
            #nn.Sequential就是按照顺序完成其中的内容
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(True),
                # 将输出的信息通道数变为out_planes
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None
    def forward(self, x):
        residual = x

        # 先批量归一化，让样本均值为0，方差为1，以便于激活函数处理
        out1 = self.bn1(x)
        #用ReLU激活函数对out处理
        out1 = F.relu(out1, True)
        #用1*1的卷积核进行卷积
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        #3*3卷积核卷积
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        #1*1卷积
        out3 = self.conv3(out3)

        #现在通道数没有满足要求就用downsample处理
        if self.downsample is not None:
            residual = self.downsample(residual)
        # 感觉是矩阵相加
        out3 += residual
        return out3

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


#上采样类，将特征图大小增加一倍
class Upsample(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Upsample, self).__init__()
        #nn.ConvTranspose2d转置卷积（反卷积）
        # 理解：直接卷积是用一个小窗户看大世界，而转置卷积是用一个大窗户的一部分去看小世界
        self.upsample = nn.ConvTranspose2d(dim_in, dim_out, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.upsample(x)


# class UpsamplePatchEmbed(nn.Module):
#     """ Image to Patch Embedding with Upsampling
#     """
#
#     def __init__(self, img_size=256, patch_size=4, stride=2, in_chans=3, embed_dim=768):
#         super().__init__()
#         self.img_size = img_size
#         self.patch_size = patch_size
#         # 反卷积层用于增加特征图的尺寸
#         self.upsample = nn.ConvTranspose2d(in_chans, in_chans, kernel_size=patch_size, stride=stride,
#                                            padding=(patch_size - stride) // 2)
#         self.norm = nn.LayerNorm(embed_dim)
#
#     def forward(self, x):
#         x = self.upsample(x)  # 使用反卷积增加特征图尺寸
#         _, _, H, W = x.shape
#         x = x.flatten(2).transpose(1, 2)  # 展平并转置
#         x = self.norm(x)  # 应用层归一化
#
#         return x, H, W

#
# @HEADS.register_module()
# class SegFormerHead(BaseDecodeHead):  # 继承了BaseDecoderHead的所有函数
#     """
#     SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
#     """
#
#     def __init__(self, feature_strides, batch_size=None, image_size=None, **kwargs):
#         super(SegFormerHead, self).__init__(input_transform='multiple_select', **kwargs)
#         # if sr_ratios is None:
#         sr_ratios = [4, 2, 1]
#         # if mlp_ratios is None:
#         mlp_ratios = [4, 4, 4]
#         # if embed_dims is None:
#         embed_dims = [256, 256, 256]
#         # if depths is None:
#         depths = [2, 2, 2, 2]
#         # if img_size is None:
#         img_size = 256
#         # if num_heads is None:
#         num_heads = [4, 4, 4]
#         drop_rate = 0.
#         attn_drop_rate = 0.
#         drop_path_rate = 0.
#         norm_layer = nn.LayerNorm
#         qkv_bias = False
#         qk_scale = None
#
#         self.depths = depths
#         self.embed_dims = embed_dims
#         # patch_embed
#         self.patch_embed1 = UpsamplePatchEmbed(img_size=img_size // 16, patch_size=4, stride=2, in_chans=embed_dims[2],
#                                                embed_dim=embed_dims[1])
#         self.patch_embed2 = UpsamplePatchEmbed(img_size=img_size // 8, patch_size=4, stride=2, in_chans=embed_dims[1],
#                                                embed_dim=embed_dims[0])
#         self.patch_embed3 = UpsamplePatchEmbed(img_size=img_size // 4, patch_size=4, stride=2, in_chans=embed_dims[0],
#                                                embed_dim=256)
#
#         # transformer decoder
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
#         cur = 0
#         self.block1 = nn.ModuleList([Block(
#             dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
#             drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
#             sr_ratio=sr_ratios[0])
#             for i in range(depths[0])])
#         self.norm1 = norm_layer(embed_dims[0])
#
#         cur += depths[0]
#         self.block2 = nn.ModuleList([Block(
#             dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
#             drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
#             sr_ratio=sr_ratios[1])
#             for i in range(depths[1])])
#         self.norm2 = norm_layer(embed_dims[1])
#
#         cur += depths[1]
#         self.block3 = nn.ModuleList([Block(
#             dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
#             drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
#             sr_ratio=sr_ratios[2])
#             for i in range(depths[2])])
#         self.norm3 = norm_layer(embed_dims[2])
#
#         self.conv1 = ConvBlock(256, 128)
#         self.upSample = Upsample(128, 128)
#         self.conv2 = nn.Conv2d(128, 68, kernel_size=3, stride=1, padding=1, bias=False)
#
#     def forward(self, inputs):
#         x = self._transform_inputs(inputs)
#         c1, c2, c3, c4 = x
#         B = c1.shape[0]
#
#         # stage1
#         # 对c4先用patchembed处理上采样，然后self-attention处理
#         x, H, W = self.patch_embed1(c4)
#         for i, blk in enumerate(self.block1):  # 这里3层block
#             x = blk(x, H, W)  # 调的是那个Block类的forward函数
#         x = self.norm1(x)
#         x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#         c3 = x + c3
#
#         # stage2
#         x, H, W = self.patch_embed2(c3)
#         for i, blk in enumerate(self.block2):  # 这里3层block
#             x = blk(x, H, W)  # 调的是那个Block类的forward函数
#         x = self.norm2(x)
#         x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#         c2 = x + c2
#
#         # stage3
#         x, H, W = self.patch_embed3(c2)
#         for i, blk in enumerate(self.block3):  # 这里3层block
#             x = blk(x, H, W)  # 调的是那个Block类的forward函数
#         x = self.norm3(x)
#         x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#         c1 = x + c1
#
#         x = self.conv1(c1)
#         x = self.upSample(x)
#         x = self.conv2(x)
#         return x

#
@HEADS.register_module()
class SegFormerHead(BaseDecodeHead):  # 继承了BaseDecoderHead的所有函数
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, batch_size=None, image_size=None, **kwargs):
        super(SegFormerHead, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim * 4,
            out_channels=embedding_dim,
            kernel_size=1,
            # norm_cfg=dict(type='SyncBN', requires_grad=True)
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, 256, kernel_size=1)
        self.conv1 = ConvBlock(256, 128)
        self.upSample = Upsample(128, 128)
        self.conv2 = nn.Conv2d(128, 124, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, inputs):
        # decoder c1-c4是encoder生成的不同尺度的特征图 为了更好地捕获语义信息
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x  # x是个由4个encoder生成的特征图的list

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape
        # linear_c4是一个MLP
        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)  # 把特征维度调整成和c1相同的尺寸

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)


        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])
        # 把四个特征在通道方向上拼接再用1*1卷积核去融合特征
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        # _c = self.linear_fuse(torch.cat([_c1], dim=1))

        x = self.dropout(_c)

        x = self.linear_pred(x)  # 最后通过这个线性层把分类结果映射回去
        x = self.conv1(x)
        x = self.upSample(x)
        x = self.conv2(x)

        return x
#
# #
# @HEADS.register_module()
# class SegFormerHead(BaseDecodeHead):
#     """
#     SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
#     """
#     def __init__(self, feature_strides, batch_size=None, image_size=None, **kwargs):
#         super(SegFormerHead, self).__init__(input_transform='multiple_select', **kwargs)
#         self.conv_up1 = ConvBlock(512, 512)
#         self.conv_up2 = ConvBlock(320, 320)
#         self.conv_up3 = ConvBlock(128, 128)
#         self.conv_up4 = ConvBlock(64, 64)
#
#         self.conv_low1 = ConvBlock(512, 512)
#         self.conv_low2 = ConvBlock(320, 320)
#         self.conv_low3 = ConvBlock(128, 128)
#         self.conv_low4 = ConvBlock(128, 128)
#
#         self.upsample1 = Upsample(512, 320)
#         self.upsample2 = Upsample(320, 128)
#         self.upsample3 = Upsample(128, 128)
#
#         self.linear_pred1 = nn.Conv2d(64, 128,kernel_size=1)
#         self.conv = ConvBlock(128, 128)
#         self.linear_pred2 = nn.Conv2d(128, 256, kernel_size=1)
#         self.conv1 = ConvBlock(256, 256)
#         self.upSample = Upsample(256, 128)
#         self.conv2 = nn.Conv2d(128, 68, kernel_size=3, stride=1, padding=1, bias=False)
#
#     def forward(self, inputs):
#         # decoder c1-c4是encoder生成的不同尺度的特征图 为了更好地捕获语义信息
#         feature = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
#         f1, f2, f3, f4 = feature
#
#         f4 = self.conv_up1(f4)
#         f3 = self.conv_up2(f3)
#         f2 = self.conv_up3(f2)
#         f1 = self.conv_up4(f1)
#
#         f4_low_in = self.conv_low1(f4)
#         f4_low = self.upsample1(f4_low_in)
#
#         f3_low_in = self.conv_low2(f3 + f4_low)
#         f3_low = self.upsample2(f3_low_in)
#
#         f2_low_in = self.conv_low3(f2 + f3_low)
#         f2_low = self.upsample3(f2_low_in)
#         f1 = self.linear_pred1(f1)
#         f1 = self.conv(f1)
#         x = self.conv_low4(f1+f2_low)
#
#         x = self.linear_pred2(x)  # 最后通过这个线性层把分类结果映射回去
#         x = self.conv1(x)
#         x = self.upSample(x)
#         x = self.conv2(x)
#
#         return x
#
