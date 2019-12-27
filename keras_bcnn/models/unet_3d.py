from __future__ import absolute_import

import numpy as np
from math import ceil, floor
from functools import partial
from keras.models import Model
from keras.layers import Conv3D, MaxPooling3D
from keras.layers import Cropping3D, Conv3DTranspose
from keras.layers.merge import Concatenate
import keras.backend as K
import tensorflow as tf

from .unet_2d import UNet2D
from ..layers import ReflectionPadding3D

_batch_axis = 0
_row_axis = 1
_col_axis = 2
_depth_axis = 3
_channel_axis = -1


class UNet3D(UNet2D):
    """ Three-dimensional U-Net

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
        dropout (bool, optional): If True, enables the dropout.
            Defaults to False.
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
                 dropout=True,
                 drop_prob=0.5,
                 pool_size=2,
                 batch_norm=False):

        super(UNet3D, self).__init__(
                 input_shape,
                 out_channels,
                 nlayer,
                 nfilter,
                 ninner,
                 3, # NOTE: temporary
                 activation,
                 conv_init,
                 upconv_init,
                 bias_init,
                 dropout,
                 drop_prob,
                 2, # NOTE: temporary
                 batch_norm)

        self._args = locals()

        # reset the kernel_size and pool_size
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        self._kernel_size = kernel_size

        if isinstance(pool_size, int):
            pool_size = (pool_size, pool_size, pool_size)
        self._pool_size = pool_size


    @property
    def pad(self):
        return ReflectionPadding3D

    @property
    def conv(self):
        return partial(Conv3D,
                        padding='valid', use_bias=True,
                        kernel_initializer=self.conv_init,
                        bias_initializer=self.bias_init)

    @property
    def upconv(self):

        return partial(Conv3DTranspose,
                        strides=self.pool_size,
                        padding='valid', use_bias=True,
                        kernel_initializer=self.upconv_init,
                        bias_initializer=self.bias_init)
    @property
    def pool(self):
        return MaxPooling3D(pool_size=self.pool_size)

    def concat(self, x1, x2):

        dx = (x1._keras_shape[_row_axis] - x2._keras_shape[_row_axis]) / 2
        dy = (x1._keras_shape[_col_axis] - x2._keras_shape[_col_axis]) / 2
        dz = (x1._keras_shape[_depth_axis] - x2._keras_shape[_depth_axis]) / 2

        crop_size = ((floor(dx), ceil(dx)), (floor(dy), ceil(dy)), (floor(dz), ceil(dz)))

        x12 = Concatenate(axis=_channel_axis)([Cropping3D(crop_size)(x1), x2])

        return x12

