from __future__ import absolute_import

from abc import ABCMeta, abstractmethod
import json


class ModelArchitect(metaclass=ABCMeta):
    """ Base class of model architecture
    """

    def save_args(self, out, args):
        args = args.copy()
        ignore_keys = ['__class__', 'self']
        for key in ignore_keys:
            if key in args.keys():
                args.pop(key)

        with open(out, 'w', encoding='utf-8') as f:
            json.dump(args, f, ensure_ascii=False, indent=4)

    @abstractmethod
    def build(self):
        raise NotImplementedError()


from .unet_2d import UNet2D
from .unet_3d import UNet3D
from .bayesian_unet_2d import BayesianUNet2D
from .bayesian_unet_3d import BayesianUNet3D
from .mc_sampler import MCSampler
