#! -*- coding: utf-8 -*-

import numpy as np

# Mean Square Error (MSE)
def mse(A, B):
    assert A.shape == B.shape, 'Shape {} is not equal to {}'.format(A.shape, B.shape)
    return np.square(A-B).mean()

# Peak Signal-to-Noise Ratio (PSNR)
def psnr(A, B, max_value=255):
    epsilon = 1e-10
    return 20 * np.log10(max_value / (np.sqrt(mse(A,B)) + epsilon))