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


from .unet import UNet
from .bayesian_unet import BayesianUNet
from .mc_sampler import MCSampler
