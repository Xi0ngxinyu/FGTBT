# 开发时间：2024/02/24 12:46
from .aflw import AFLW
from .cofw import COFW
from .face300W import Face300W
from .wflw import WFLW
from .mix_dataset import MixedDataset
from .transforms import get_transformer_coords
# import transforms

__all__ = ['AFLW', 'COFW', 'Face300W', 'WFLW', 'mix_dataset','get_dataset']


def get_dataset(config):
    if config['dataset']['dataset'] == 'AFLW':
        return AFLW
    elif config['dataset']['dataset'] == 'COFW':
        return COFW
    elif config['dataset']['dataset'] == '300W':
        return Face300W
    elif config['dataset']['dataset'] == 'WFLW':
        return WFLW
    else:
        raise NotImplemented()
