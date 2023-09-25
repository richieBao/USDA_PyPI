# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 15:11:00 2022

@author: Richie Bao-caDesign设计(cadesign.cn)
"""
from .random_walk import temporal_difference
from .random_walk import monte_carlo
from .random_walk import compute_state_value
from .random_walk import rms_error
from .random_walk import batch_updating
from .random_walk import example_6_2
from .random_walk import figure_6_2

from .windy_grid_world import figure_6_3
from .cliff_walking import figure_6_4
from .cliff_walking import figure_6_6

__all__=[
    "temporal_difference",
    "monte_carlo",
    "compute_state_value",
    "rms_error",
    "batch_updating",
    "example_6_2",
    "figure_6_2",
    "figure_6_3",
    "figure_6_4",
    "figure_6_6",
    ]
