from __future__ import absolute_import

import tensorflow as tf
from keras.layers import ZeroPadding2D
from keras.backend.tensorflow_backend import normalize_data_format
from keras.backend.tensorflow_backend import transpose_shape


def spatial_2d_padding(x, padding=((1, 1), (1, 1)), mode='REFLECT', data_format=None):
    """ Pads the 2nd and 3rd dimensions of a 4D tensor.

    # Arguments
        x: Tensor or variable.
        padding: Tuple of 2 tuples, padding pattern.
        data_format: string, `"channels_last"` or `"channels_first"`.

    # Returns
        A padded 4D tensor.

    # Raises
        ValueError: if `data_format` is neither `"channels_last"` or `"channels_first"`.
    """
    assert len(padding) == 2
    assert len(padding[0]) == 2
    assert len(padding[1]) == 2
    data_format = normalize_data_format(data_format)

    pattern = [[0, 0],
               list(padding[0]),
               list(padding[1]),
               [0, 0]]
    pattern = transpose_shape(pattern, data_format, spatial_axes=(1, 2))
    return tf.pad(x, pattern, mode)


class ReflectionPadding2D(ZeroPadding2D):

    def call(self, inputs):
        return spatial_2d_padding(inputs,
                                  padding=self.padding,
                                  data_format=self.data_format)
