import os
import cv2
import matplotlib.pyplot as plt
from keras.utils import plot_model

from keras_bcnn.models import UNet2D
from keras_bcnn.models import BayesianUNet2D
from keras_bcnn.models import UNet3D
from keras_bcnn.models import BayesianUNet3D
from keras_bcnn.models import MCSampler


def test_2d():

    input_shape = (512,512,1)

    archit = BayesianUNet2D(input_shape, 23, nlayer=5)
    model = archit.build()
    model.summary()
    plot_model(model, to_file='bayesian_unet_2d.png', show_shapes=True)

    predictor = MCSampler(model, mc_iteration=10).build()
    predictor.summary()
    plot_model(predictor, to_file='mc_sampler_2d.png', show_shapes=True)


def test_3d():

    input_shape = (128,128,128,1)

    archit = BayesianUNet3D(input_shape, 23, nlayer=3)
    model = archit.build()
    model.summary()
    plot_model(model, to_file='bayesian_unet_3d.png', show_shapes=True)

    predictor = MCSampler(model, mc_iteration=10).build()
    predictor.summary()
    plot_model(predictor, to_file='mc_sampler_3d.png', show_shapes=True)


if __name__ == '__main__':

    test_2d()
    test_3d()
