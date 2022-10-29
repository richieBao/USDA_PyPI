# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 15:11:00 2022

@author: Richie Bao-caDesign设计(cadesign.cn)
"""
from ._neighbors import k_neighbors_entire
from ._computational_performance import PolynomialFeatures_regularization

from ._dim1_convolution import dim1_convolution_SubplotAnimation
from ._dim1_convolution import G_T_type_1
from ._dim1_convolution import F_T_type_1
from ._dim1_convolution import G_T_type_2
from ._dim1_convolution import G_T_type_2
from ._curve_segmentation import curve_segmentation_1DConvolution
from ._sir_model import SIR_deriv
from ._sir_model import convolution_diffusion_img
from ._sir_model import SIR_spatialPropagating

__all__=[
    "k_neighbors_entire",
    "PolynomialFeatures_regularization",
    "dim1_convolution_SubplotAnimation",
    "G_T_type_1",
    "F_T_type_1",
    "G_T_type_2",
    "G_T_type_2",
    "curve_segmentation_1DConvolution"
    "SIR_deriv",
    "convolution_diffusion_img",
    ]



