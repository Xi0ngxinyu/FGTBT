import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from utils import create_logger
import models
from test import validate
from datasets import COFW, WFLW, Face300W, AFLW
from yacs.config import CfgNode as CN
import glob
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def make_data_loaders(config):
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
    return val_loaders


def test_ckpt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/face_alignment_300W.yaml')
    args = parser.parse_args()
    with open(args.config, 'r', encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = CN(config)
    val_loaders = make_data_loaders(config)

    folder_path = r"D:\learn\BABTEMA4"
    pth_files = glob.glob(os.path.join(folder_path, '*.pth'))
    log = create_logger(config)
    for file_path in pth_files:
        print("test: "+file_path)
        sv_file = torch.load(file_path)
        state_dict = sv_file['model']['sd']
        model = models.make(config['model'], state_dict=state_dict, load_sd=True).cuda()
        epoch = sv_file['epoch']

        log_info = ['Epoch: [{0}]'.format(epoch)]

        nme_ip_300w, nme_io_300w, _ = validate(config, val_loaders[2], model)
        log_info.append('nme_ip_300w={:.5f}, nme_io_300w={:.5f}'.format(nme_ip_300w, nme_io_300w))
        print('nme_ip_300w={:.5f}, nme_io_300w={:.5f}'.format(nme_ip_300w, nme_io_300w))
        # nme_ip_aflw, nme_io_aflw, _ = validate(config, val_loaders[0], model)
        # log_info.append('nme_ip_aflw={:.5f}, nme_io_aflw={:.5f}'.format(nme_ip_aflw, nme_io_aflw))
        # print('nme_ip_aflw={:.5f}, nme_io_aflw={:.5f}'.format(nme_ip_aflw, nme_io_aflw))
        # nme_ip_wflw, nme_io_wflw, _ = validate(config, val_loaders[1], model)
        # log_info.append('nme_ip_wflw={:.5f}, nme_io_wflw={:.5f}'.format(nme_ip_wflw, nme_io_wflw))
        # print('nme_ip_wflw={:.5f}, nme_io_wflw={:.5f}'.format(nme_ip_wflw, nme_io_wflw))
        # nme_ip_cofw, nme_io_cofw, failure_010_rate_cofw = validate(config, val_loaders[3], model)
        # log_info.append('nme_ip_cofw={:.5f}, nme_io_cofw={:.5f}, failure_010_rate_cofw={:.5f}'
        #                 .format(nme_ip_cofw, nme_io_cofw, failure_010_rate_cofw))
        # print('nme_ip_cofw={:.5f}, nme_io_cofw={:.5f}, failure_010_rate_cofw'
        #       .format(nme_ip_cofw, nme_io_cofw, failure_010_rate_cofw))
        log.info(', '.join(log_info))


if __name__ == '__main__':
    test_ckpt()
