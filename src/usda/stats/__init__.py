# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 15:11:00 2022

@author: Richie Bao-caDesign设计(cadesign.cn)
"""
from ._descriptive_stats import frequency_bins
from ._descriptive_stats import comparisonOFdistribution

from ._outlier import is_outlier
from ._kde import ptsKDE_geoDF2raster

from ._regression import coefficient_of_determination
from ._regression import ANOVA
from ._regression import confidenceInterval_estimator_LR
from ._regression import correlationAnalysis_multivarialbe
from ._regression import coefficient_of_determination_correction
from ._regression import confidenceInterval_estimator_LR_multivariable

__all__ = [
    "frequency_bins",
    "comparisonOFdistribution",
    "is_outlier",
    "ptsKDE_geoDF2raster",
    "coefficient_of_determination",
    "ANOVA",
    "confidenceInterval_estimator_LR",
    "correlationAnalysis_multivarialbe",
    "coefficient_of_determination_correction",
    "confidenceInterval_estimator_LR_multivariable",
    "P_Xk_binomialDistribution",
    ]

