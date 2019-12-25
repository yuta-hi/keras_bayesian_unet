# Bayesian U-Net for Keras

This is a Keras re-implementation for [Bayesian U-Net](https://github.com/yuta-hi/bayesian_unet)
Currently, this repository only supports 2D...

<img src='https://github.com/yuta-hi/bayesian_unet/blob/master/figs/bayesian_unet.gif' width='400px'>

## Requirements
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN
- Keras and TensorFlow

## Installation
- Install from this repository
```bash
git clone https://github.com/yuta-hi/keras_bayesian_unet
cd keras_bayesian_unet
pip install .
```

## Usage
- Bayesian U-Net
```python
from keras_bcnn.models import BayesianUNet
input_shape = (512, 512, 1)
output_channles = 23
model = BayesianUNet(input_shape, output_channles).build()
```

- MC sampler
```python
from keras_bcnn.models import MCSampler
mc_iteration = 10
sampler = MCSampler(model, mc_iteration).build()
```

## Related repositories
- [bayesian_unet](https://github.com/yuta-hi/bayesian_unet)

- [anatomy_viewer](https://github.com/yuta-hi/anatomy-viewer)
<img src='https://github.com/yuta-hi/anatomy-viewer/blob/master/figs/demo.gif' width='600px'>
