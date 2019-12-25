import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import floor, ceil
from keras.layers import Input
from keras.layers import Cropping2D
from keras.models import Model
from keras.utils import plot_model
import keras.backend as K

from keras_bcnn.models import UNet
from keras_bcnn.models import BayesianUNet
from keras_bcnn.models import MCSampler
from keras_bcnn.initializers import bilinear_upsample

from keras.layers import UpSampling2D

_row_axis = 1
_col_axis = 2

class TestBilinerUpSample(UNet):

    def build(self):
        inputs = Input(self.input_shape)
        h = self.pad(1)(inputs)
        h = self.upconv(self.out_channels, 3)(h)

        dx = (h._keras_shape[_row_axis] - inputs._keras_shape[_row_axis]*2) / 2
        dy = (h._keras_shape[_col_axis] - inputs._keras_shape[_col_axis]*2) / 2

        crop_size = ((floor(dx), ceil(dx)), (floor(dy), ceil(dy)))

        outputs = Cropping2D(crop_size)(h)

        return Model(inputs=inputs, outputs=outputs)

class TestUpSample(UNet):

    @property
    def upconv(self):
        return UpSampling2D()

    def build(self):
        inputs = Input(self.input_shape)
        outputs = self.upconv(inputs)
        return Model(inputs=inputs, outputs=outputs)

def test():

    test_image = 'lenna.png'

    image = cv2.imread(test_image)
    image = cv2.resize(image, (64,64))
    image = image[np.newaxis].astype(np.float32)

    x = image

    archit = TestBilinerUpSample(
                image.shape[1:],
                image.shape[-1],
                upconv_init=bilinear_upsample())
    model = archit.build()
    model.summary()

    y_upconv = model(K.variable(x))

    archit = TestUpSample(
                image.shape[1:],
                image.shape[-1])
    model = archit.build()
    model.summary()

    y_upsample = model(K.variable(x))

    plt.subplot(1,3,1)
    plt.imshow(x[0,:,:,0], cmap='gray')
    plt.title('x')
    plt.colorbar()
    plt.subplot(1,3,2)
    plt.imshow(K.eval(y_upsample)[0,:,:,0], cmap='gray')
    plt.title('upsample(x)')
    plt.colorbar()
    plt.subplot(1,3,3)
    plt.imshow(K.eval(y_upconv)[0,:,:,0], cmap='gray')
    plt.title('upconv(x)')
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    test()
