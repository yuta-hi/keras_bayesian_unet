import os
import cv2
import matplotlib.pyplot as plt
from keras.utils import plot_model

from keras_bcnn.models import UNet
from keras_bcnn.models import BayesianUNet
from keras_bcnn.models import MCSampler


if __name__ == '__main__':

    input_shape = (512,512,1)

    archit = BayesianUNet(input_shape, 23, nlayer=5)
    model = archit.build()
    model.summary()
    plot_model(model, to_file='bayesian_unet.png', show_shapes=True)

    predictor = MCSampler(model, mc_iteration=10).build()
    predictor.summary()
    plot_model(predictor, to_file='mc_sampler.png', show_shapes=True)
