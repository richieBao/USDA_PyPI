# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 15:11:00 2022

@author: Richie Bao-caDesign设计(cadesign.cn)
"""
from ._table_show import plotly_table
from ._table_show import print_html

from ._stats_charts import probability_graph
from ._stats_charts import demo_con_style

from ._colors import generate_colors


__all__ = [
    "plotly_table",
    "print_html",
    "probability_graph",
    "generate_colors",
    ]


