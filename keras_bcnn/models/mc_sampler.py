from __future__ import absolute_import

from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Lambda
from keras.layers import Reshape
import keras.backend as K
import tensorflow as tf

from . import ModelArchitect

_batch_axis = 0
_channel_axis = -1

_reduce_table = {
    'none': lambda x: x,
    'mean': K.mean,
    'std': K.std,
    'var': K.var,
    'argmax': K.argmax,
    'argmin': K.argmin,
}

class MCSampler(ModelArchitect):
    """ Monte Carlo estimation to approximate the predictive distribution.
    Predictive variance is a metric indicating uncertainty.
    Args:
        predictor (~keras.models.Model): Predictor network.
        mc_iteration (int): Number of iterations in MCMC sampling
        activation (str, optional): Activation function.
            Defaults to 'softmax'.
        reduce_mean (str, optional): Reduce function along the channel axis for mean tensor.
            Defaults to 'argmax'.
        reduce_var (str, optional): Reduce function along the channel axis for variance tensor.
            Defaults to 'mean'.

    Note:
        Default values ​​are set assuming segmentation task.

    See also: https://arxiv.org/pdf/1506.02142.pdf
              https://arxiv.org/pdf/1511.02680.pdf
    """
    def __init__(self,
                 predictor,
                 mc_iteration,
                 activation='softmax',
                 reduce_mean='argmax',
                 reduce_var='mean'):

        self._predictor = predictor
        self._mc_iteration = mc_iteration
        self._activation = activation
        self._reduce_mean = reduce_mean
        self._reduce_var = reduce_var

        input_shape = predictor.layers[0].input_shape[1:]
        self._input_shape = input_shape


    @property
    def input_shape(self):
        return self._input_shape

    @property
    def predictor(self):
        return self._predictor
    @property
    def mc_iteration(self):
        return self._mc_iteration

    @property
    def activation(self):
        if self._activation is not None:
            return Activation(self._activation)
        else:
            return lambda x: x

    @property
    def reduce_mean(self):
        return _reduce_table[self._reduce_mean]

    @property
    def reduce_var(self):
        return _reduce_table[self._reduce_var]

    def build(self):

        inputs = Input(self.input_shape)
        mc_samples = Lambda(lambda x: K.repeat_elements(x, self.mc_iteration, axis=_batch_axis))(inputs)

        logits = self.predictor(mc_samples)
        probs  = self.activation(logits)

        ret_shape = self.predictor.layers[-1].output_shape
        ret_shape = (-1, self.mc_iteration, *ret_shape[1:])

        probs = Lambda(lambda x: K.reshape(x, ret_shape))(probs)

        mean = Lambda(lambda x: K.mean(x, axis=1))(probs)
        mean = Lambda(lambda x: self.reduce_mean(x, axis=_channel_axis))(mean)

        variance = Lambda(lambda x: K.var(x, axis=1))(probs)
        variance = Lambda(lambda x: self.reduce_var(x, axis=_channel_axis))(variance)

        return Model(inputs=inputs, outputs=[mean, variance])
