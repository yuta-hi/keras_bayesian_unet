import numpy as np
import tensorflow as tf
import chainer

from keras_bcnn.losses import softmax_cross_entropy

def test():

    b, w, h, c = 5, 20, 30, 10

    x = tf.placeholder(tf.float32, [None, w, h, c])
    t = tf.placeholder(tf.int32, [None, w, h])

    loss = softmax_cross_entropy(t, x)

    sess = tf.Session()

    _x = np.random.rand(b, w, h, c).astype(np.float32)
    _t = np.random.randint(0, c, (b, w, h)).astype(np.int32)

    ret = sess.run(loss, feed_dict={x: _x, t: _t})
    print(ret)

    _x = _x.transpose(0, 3, 1, 2) # NOTE: convert to chainer's format
    ret = chainer.functions.softmax_cross_entropy(_x, _t, normalize=False)
    print(ret.data)


if __name__ == '__main__':
    test()
