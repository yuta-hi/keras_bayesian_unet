#!/usr/bin/env python

from setuptools import find_packages
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='keras_bcnn',
    version='1.0.0',
    description='Bayesian U-Net for Keras',
    long_description=open('README.md').read(),
    author='yuta-hi',
    packages=find_packages(),
    include_package_data=True,
    install_requires=open('requirements.txt').readlines(),
    url='https://github.com/yuta-hi/keras_bayesian_unet',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)
