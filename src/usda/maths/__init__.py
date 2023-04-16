# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 15:11:00 2022

@author: Richie Bao-caDesign设计(cadesign.cn)
"""
from ._algebra import vector_plot_2d
from ._algebra import vector_plot_3d
from ._algebra import move_alongVectors
from ._algebra import vector2matrix_rref

from ._geometric_calculation import circle_lines
from ._geometric_calculation import point_Proj2Line

from ._plot_single_function import plot_single_function

__all__=[
    "vector_plot_2d",
    "vector_plot_3d",
    "move_alongVectors",
    "vector2matrix_rref",
    "circle_lines",
    "point_Proj2Line",
    "plot_single_function",
    ]



