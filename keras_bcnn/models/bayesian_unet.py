from __future__ import absolute_import

from functools import partial

from .unet import UNet
from ..layers import MCDropout


class BayesianUNet(UNet):
    """ Bayesian U-Net

    Args:
        input_shape (list): Shape of an input tensor.
        out_channels (int): Number of output channels.
        nlayer (int, optional): Number of layers.
            Defaults to 5.
        nfilter (list or int, optional): Number of filters.
            Defaults to 32.
        ninner (list or int, optional): Number of layers in UNetBlock.
            Defaults to 2.
        kernel_size (list or int): Size of convolutional kernel. Defaults to 3.
        activation (str, optional): Type of activation layer.
            Defaults to 'relu'.
        conv_init (str, optional): Type of kernel initializer for conv. layer.
            Defaults to 'he_normal'.
        upconv_init (str, optional): Type of kernel initializer for upconv. layer.
            Defaults to 'he_normal'.
        bias_init (str, optional): Type of bias initializer for conv. and upconv. layer.
        drop_prob (float, optional): Ratio of dropout.
            Defaults to 0.5.
        pool_size (int, optional): Size of spatial pooling.
            Defaults to 2.
        batch_norm (bool, optional): If True, enables the batch normalization.
            Defaults to False.
    """
    def __init__(self,
                 input_shape,
                 out_channels,
                 nlayer=5,
                 nfilter=32,
                 ninner=2,
                 kernel_size=3,
                 activation='relu',
                 conv_init='he_normal',
                 upconv_init='he_normal',
                 bias_init='zeros',
                 drop_prob=0.5,
                 pool_size=2,
                 batch_norm=False):

        args = locals()
        ignore_keys = ['__class__', 'self']
        for key in ignore_keys:
            if key in args.keys():
                args.pop(key)
        args['dropout'] = True # NOTE: force
        super().__init__(**args)

    @property
    def dropout(self):
        if self._dropout:
            return partial(MCDropout(self._drop_prob))
        else:
            raise ValueError()

