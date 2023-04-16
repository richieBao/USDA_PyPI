# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 13:07:22 2023

@author: richie bao
"""
from torch.nn import AvgPool2d
from torch.nn.functional import interpolate
import numpy as np

def progressive_down_sampling(real_batch, total_depth,current_depth, alpha):
    """
    private helper for down_sampling the original images in order to facilitate the
    progressive growing of the layers.

    :param real_batch: batch of real samples
    :param depth: depth at which training is going on
    :param alpha: current value of the fade-in alpha
    :return: real_samples => modified real batch of samples
    """

    # down_sample the real_batch for the given depth
    down_sample_factor = int(np.power(2,total_depth - current_depth- 1))
    prior_down_sample_factor = max(int(np.power(2, total_depth - current_depth)), 0)

    ds_real_samples = AvgPool2d(down_sample_factor)(real_batch)

    if current_depth > 0:
        prior_ds_real_samples = interpolate(AvgPool2d(prior_down_sample_factor)(real_batch), scale_factor=2)
    else:
        prior_ds_real_samples = ds_real_samples

    # real samples are a combination of ds_real_samples and prior_ds_real_samples
    real_samples = (alpha * ds_real_samples) + ((1 - alpha) * prior_ds_real_samples)

    # return the so computed real_samples
    return real_samples