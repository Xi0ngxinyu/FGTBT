import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class MixedDataset(Dataset):
    def __init__(self, datasets):
        """
        :param datasets: 一个包含多个 DataLoader 或 Dataset 对象的列表
        """
        self.datasets = datasets
        self.data_iters = [iter(dataset) for dataset in datasets]  # 创建每个数据集的迭代器

    def __len__(self):
        # 返回合并后的数据集长度，这里假设所有数据集的长度相同，选择最小长度作为合并后的长度
        return min(len(dataset) for dataset in self.datasets)

    def __getitem__(self, idx):
        """
        从每个数据集中获取一个样本，合并并返回。
        :param idx: 索引
        """
        images_mix = []
        target_mix = []
        target_heatmap_mix = []
        meta_mix = []

        # 从每个数据加载器中获取一个批次数据
        for ii in range(len(self.datasets)):
            try:
                img, target, target_heatmap, meta = next(self.data_iters[ii])
            except StopIteration:
                # 如果迭代器耗尽，重新初始化迭代器
                self.data_iters[ii] = iter(self.datasets[ii])
                img, target, target_heatmap, meta = next(self.data_iters[ii])

            images_mix.append(img)
            target_mix.append(target)
            target_heatmap_mix.append(target_heatmap)
            meta_mix.append(meta)

        # 将每个数据加载器的批次合并成一个大的批次
        images_mix = torch.cat(images_mix, dim=0)
        # target_mix = torch.cat(target_mix, dim=0)
        # target_heatmap_mix = torch.cat(target_heatmap_mix, dim=0)
        # meta_mix = meta_mix  # meta 信息如果需要合并，可以再进行处理

        return images_mix, target_mix, target_heatmap_mix, meta_mix

