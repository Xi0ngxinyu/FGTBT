import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import yaml
import torch
import torch.nn as nn
import pprint
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR, CosineAnnealingLR, StepLR
from utils import create_logger

import models
import utils
from test import validate

from datasets import COFW, WFLW, Face300W, AFLW
from datasets.mix_dataset import MixedDataset
from yacs.config import CfgNode as CN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
import cv2
import matplotlib.pyplot as plt


def visualize_with_landmarks(images, targets):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    batch_size = images.shape[0]
    for i in range(batch_size):
        image = images[i]
        landmarks = targets[i]

        # 将 PyTorch 张量转换为 NumPy 数组
        image_np = image.cpu().numpy()
        image_np = np.transpose(image_np, (1, 2, 0))  # 从 [C, H, W] 转换为 [H, W, C]

        # 反标准化图像
        image_np = (image_np * std + mean) * 255.0
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)  # 将图像裁剪到 [0, 255] 范围并转换为 uint8

        # 将图像从 RGB 转换为 BGR（OpenCV 使用 BGR 顺序）
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # 创建图像的副本，以免修改原始图像
        image_with_landmarks = np.ascontiguousarray(image_bgr)

        # 将特征点绘制到图像上
        for landmark in landmarks:
            x, y = landmark[0]*2, landmark[1]*2
            cv2.circle(image_with_landmarks, (int(x), int(y)), radius=1, color=(0, 255, 0), thickness=-1)  # 绿色

        # 显示带有特征点的图像
        plt.imshow(cv2.cvtColor(image_with_landmarks, cv2.COLOR_BGR2RGB))
        plt.show()


# def custom_collate_fn(batch):
#     images_mix, target_mix, target_heatmap_mix, meta_mix = batch[0]
#     return images_mix, target_mix, target_heatmap_mix, meta_mix


def accumulate_net(model1, model2, decay):
    """
        operation: model1 = model1 * decay + model2 * (1 - decay)
    """
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())
    for k in par1.keys():
        par1[k].data.mul_(decay).add_(
            other=par2[k].data.to(par1[k].data.device),
            alpha=1 - decay)

    par1 = dict(model1.named_buffers())
    par2 = dict(model2.named_buffers())
    for k in par1.keys():
        if par1[k].data.is_floating_point():
            par1[k].data.mul_(decay).add_(
                other=par2[k].data.to(par1[k].data.device),
                alpha=1 - decay)
        else:
            par1[k].data = par2[k].data.to(par1[k].data.device)


def make_data_loaders():
    # train_loader = make_data_loader(config, tag='train')
    # val_loader = make_data_loader(config, tag='val')
    cofw_train_loader = DataLoader(
        dataset=COFW(config, is_train=True),
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY)
    wflw_train_loader = DataLoader(
        dataset=WFLW(config, is_train=True),
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY)
    face300w_train_loader = DataLoader(
        dataset=Face300W(config, is_train=True),
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY)
    aflw_train_loader = DataLoader(
        dataset=AFLW(config, is_train=True),
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY)
    # 使用混合数据集
    train_loaders = [cofw_train_loader, wflw_train_loader, face300w_train_loader, aflw_train_loader]
    # mixed_dataset = MixedDataset(train_datasets)
    # train_loader = DataLoader(
    #     dataset=mixed_dataset,
    #     batch_size=1,
    #     shuffle=config.TRAIN.SHUFFLE,
    #     num_workers=config.WORKERS,
    #     pin_memory=config.PIN_MEMORY,
    #     collate_fn=custom_collate_fn
    # )
    aflw_val_loader = DataLoader(
        dataset=AFLW(config, is_train=False),
        batch_size=config.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY)
    face300w_val_loader = DataLoader(
        dataset=Face300W(config, is_train=False),
        batch_size=config.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    wflw_val_loader = DataLoader(
        dataset=WFLW(config, is_train=False),
        batch_size=config.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    cofw_val_loader = DataLoader(
        dataset=COFW(config, is_train=False),
        batch_size=config.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY)

    val_loaders = [aflw_val_loader, wflw_val_loader, face300w_val_loader, cofw_val_loader]
    return train_loaders, val_loaders


# 自定义学习率调度函数
# def lr_lambda(step):
#     if step < 80001:
#         return 1.0  # 保持原始学习率
#     elif step < 120000:
#         return 0.2  # 80,000 下降到 0.2 倍
#     else:
#         return 0.1  # 120,000 下降到 0.5 倍


def prepare_training():
    if config.get('resume') is not None:
        sv_file = torch.load(config['resume'])
        state_dict = sv_file['model']['sd']
        model = models.make(config['model'], state_dict=state_dict, load_sd=True).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        epoch_start = sv_file['epoch'] + 1
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
    # milestones = [20, 40, 60]  # 设置学习率下降的位置
    lr_scheduler = MultiStepLR(optimizer, milestones=config['milestones'], gamma=0.8)

    # lr_scheduler= LambdaLR(optimizer, lr_lambda=lr_lambda)
    # lr_scheduler = CosineAnnealingLR(optimizer, config['epoch_max'], eta_min=8e-5)
    log.info('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    model.to(device)
    return model, optimizer, epoch_start, lr_scheduler


def main(config_, save_path):
    torch.autograd.set_detect_anomaly(True)
    global config, log, writer
    config = config_  # 一个dict 从yaml文件里读的
    log = create_logger(cfg)
    log.info(pprint.pformat(config))

    train_loader, val_loaders = make_data_loaders()

    model, optimizer, epoch_start, lr_scheduler = prepare_training()  # 加载

    if config.ema:
        model_ema = models.make(config['model']).cuda()
        model_ema.eval()
        accumulate_net(model_ema, model, 0)
    else:
        model_ema = None

    model.set_landmark_index(config.DATASET_COFW.LANDMARK_INDEX, config.DATASET_AFLW.LANDMARK_INDEX,
                             config.DATASET_300W.LANDMARK_INDEX, config.DATASET_WFLW.LANDMARK_INDEX,
                             config.WEIGHT, config.beta)
    model.optimizer = optimizer

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    epoch_max = config['epoch_max']  #
    epoch_val = config.get('epoch_val')  # 1
    epoch_save = config.get('epoch_save')
    min_val_300w = 1e8
    min_val_aflw = 1e8
    min_val_wflw = 1e8
    min_val_cofw = 1e8

    timer = utils.Timer()  # 一个用来得到时间的对象
    #######################
    for param in model.parameters():
        param.requires_grad = True
    #################################
    # 训练
    # 初始化所有的data_iters
    data_iters = []
    for i in range(4):
        data_iters.append(iter(train_loader[i]))
    t_epoch_start = timer.t()
    loss = utils.Averager()
    for epoch in range(epoch_start, epoch_max + 1):
        # print(epoch)
        images_mix = []
        target_mix = []
        target_heatmap_mix = []
        meta_mix = []
        for ii in range(4):
            try:
                img, target, target_heatmap, meta = next(data_iters[ii])
            except StopIteration:
                # 如果迭代器耗尽，重新初始化迭代器
                data_iters[ii] = iter(train_loader[ii])
                img, target, target_heatmap, meta = next(data_iters[ii])
            images_mix.append(img)
            target_mix.append(target)
            target_heatmap_mix.append(target_heatmap)
            meta_mix.append(meta)

            # visualize_with_landmarks(img, target)

        # del img, target, target_heatmap, meta
        # torch.cuda.empty_cache()
        model.set_input(images_mix, target_mix, target_heatmap_mix, config.TRAIN.BATCH_SIZE)
        model.optimize_parameters()
        loss.add(model.loss_G.item())
        lr_scheduler.step()
        # ema
        if model_ema is not None:
            accumulate_net(model_ema, model, 0.5 ** (config.TRAIN.BATCH_SIZE*4 / 10000.0))

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            log_info = ['Epoch: [{0}/{1}] lr:{lr:}, train:loss:{loss:}'
                        .format(epoch, epoch_max, lr=lr_scheduler.get_last_lr()[0], loss=loss.item())]
            loss.clear()
            if n_gpus > 1 and (config.get('eval_bsize') is not None):
                model_ = model.module
            else:
                model_ = model

            if n_gpus > 1 and (config.get('eval_bsize') is not None):
                model__ = model_ema.module
            else:
                model__ = model_ema
            # 测试
            #  ##################测试函数#######################  #
            nme_ip_300w, nme_io_300w, _ = validate(config, val_loaders[2], model_)
            log_info.append('nme_ip_300w={:.5f}, nme_io_300w={:.5f}'.format(nme_ip_300w, nme_io_300w))
            nme_ip_300w_ema, nme_io_300w_ema, _ = validate(config, val_loaders[2], model__)
            log_info.append('nme_ip_300w_ema={:.5f}, nme_io_300w_ema={:.5f}'.format(nme_ip_300w_ema, nme_io_300w_ema))
            if nme_io_300w_ema < 0.046 and nme_io_300w_ema < min_val_300w:
                min_val_300w = nme_io_300w_ema
                save(config, model_ema, optimizer, save_path, 'best_300w', 0)

            if epoch % 5000 == 0:
                nme_ip_aflw, nme_io_aflw, _ = validate(config, val_loaders[0], model__)
                log_info.append('nme_ip_aflw={:.5f}, nme_io_aflw={:.5f}'.format(nme_ip_aflw, nme_io_aflw))
                nme_ip_wflw, nme_io_wflw, _ = validate(config, val_loaders[1], model__)
                log_info.append('nme_ip_wflw={:.5f}, nme_io_wflw={:.5f}'.format(nme_ip_wflw, nme_io_wflw))
                nme_ip_cofw, nme_io_cofw, failure_010_rate_cofw = validate(config, val_loaders[3], model__)
                log_info.append('nme_ip_cofw={:.5f}, nme_io_cofw={:.5f}, failure_010_rate_cofw={:.5f}'
                                .format(nme_ip_cofw, nme_io_cofw, failure_010_rate_cofw))
                if nme_io_aflw < 0.017 and nme_io_aflw < min_val_aflw:
                    min_val_aflw = nme_io_aflw
                    save(config, model_ema, optimizer, save_path, 'best_aflw', 0)
                if nme_io_wflw < 0.0455 and nme_io_wflw < min_val_wflw:
                    min_val_wflw = nme_io_wflw
                    save(config, model_ema, optimizer, save_path, 'best_wflw', 0)
                if nme_ip_cofw < 0.049 and nme_ip_cofw < min_val_cofw:
                    min_val_cofw = nme_ip_cofw
                    save(config, model_ema, optimizer, save_path, 'best_cofw', 0)

            t = timer.t()
            prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
            t_epoch = utils.time_text(t - t_epoch_start)
            t_epoch_start = timer.t()
            t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
            log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

            log.info(', '.join(log_info))

        if epoch % epoch_save == 0:
            save(config, model_ema, optimizer, save_path, '', epoch)


def save(config, model, optimizer, save_path, name,epoch, save_optimizer=True):
    """
    保存checkpoint，包含模型的名称、参数配置以及状态字典，同时（可选）保存优化器状态和当前epoch。

    :param config: 模型配置字典，要求至少包含 config['model']['name'] 和 config['model']['args']
    :param model: 需要保存的模型
    :param optimizer: 训练使用的优化器
    :param save_path: 保存checkpoint的目录
    :param epoch: 当前训练的epoch或其他标识信息
    :param save_optimizer: 是否保存优化器状态（默认True）
    """
    os.makedirs(save_path, exist_ok=True)
    model_spec = {
        'name': config['model']['name'],
        'args': config['model']['args'],
        'sd': model.state_dict()
    }
    checkpoint = {
        'model': model_spec,
        'epoch': epoch
    }
    if save_optimizer:
        optimizer_spec = {
            'name': config['optimizer']['name'],
            'args': config['optimizer']['args'],
            'sd': optimizer.state_dict()
        }
        checkpoint['optimizer'] = optimizer_spec
    save_file = os.path.join(save_path, f"full_model_epoch{epoch}_{name}.pth")
    torch.save(checkpoint, save_file)
    print(f"Checkpoint已保存至: {save_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/face_alignment_300W.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r',encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    # 将字典转换为CfgNode对象
    cfg = CN(config)
    save_name = args.name
    # if save_name is None:
    #     save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    # if args.tag is not None:
    #     save_name += '_' + args.tag
    save_path = os.path.join('./save')

    main(cfg, save_path)
