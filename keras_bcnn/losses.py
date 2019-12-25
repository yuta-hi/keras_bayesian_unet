from __future__ import absolute_import

import tensorflow as tf
from keras.backend.tensorflow_backend import _to_tensor, epsilon, cast
import warnings

def _check_dtype(tensor, dtype):
    if tensor.dtype.base_dtype != dtype:
        comment = 'The dtype of target should be'
        warnings.warn('%s %s != %s' % (comment, dtype, tensor.dtype.base_dtype))

def softmax_cross_entropy(target, output, from_logits=True, axis=-1, normalize=False):
    """ Compute Softmax cross entropy loss for sparse target.

    Args:
        target (tensor): Target label. If 2D, shape is (w, h).
        output (tensor): Logits or Probabilities. If 2D, shape is (w, h, ch).
        from_logits (bool, optional): logits or softmax outputs? Defaults to True.
        axis (int, optional): Specifying the channels axis. Defaults to -1.
        normalize (bool, optional): Normalize loss across all instances. Defaults to False.
    """
    _check_dtype(target, 'int32')
    _check_dtype(output, 'float32')

    output_dimensions = list(range(len(output.get_shape())))
    if axis != -1 and axis not in output_dimensions:
        raise ValueError(
            '{}{}{}'.format(
                'Unexpected channels axis {}. '.format(axis),
                'Expected to be -1 or one of the axes of `output`, ',
                'which has {} dimensions.'.format(len(output.get_shape()))))

    # move the channels to be in the last axis:
    if axis != -1 and axis != output_dimensions[-1]:
        permutation = output_dimensions[:axis] + output_dimensions[axis + 1:]
        permutation += [axis]
        output = tf.transpose(output, perm=permutation)

    # convert to the logits
    if not from_logits:
        _epsilon = _to_tensor(epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
        output = tf.log(output) # NOTE: log(exp(x)) = x
    logits = output

    # softmax_cross_entropy
    output_shape = output.get_shape()
    targets = cast(tf.reshape(target, tf.shape(output)[:-1]), 'int32') # NOTE: cast...

    res = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=targets,
        logits=logits)

    # reduce
    if normalize:
        return tf.reduce_mean(res)
    else:
        return tf.reduce_sum(tf.reduce_mean(res, axis=0)) # only batch-axis
