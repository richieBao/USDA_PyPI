# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 15:11:00 2022

@author: Richie Bao-caDesign设计(cadesign.cn)
"""
from .blackjack import play
from .blackjack import monte_carlo_on_policy
from .blackjack import monte_carlo_es
from .blackjack import monte_carlo_off_policy
from .blackjack import figure_5_1
from .blackjack import figure_5_2
from .blackjack import figure_5_3

__all__=[
    "play",
    "monte_carlo_on_policy",
    "monte_carlo_es",
    "monte_carlo_off_policy",
    "figure_5_1",
    "figure_5_2",
    "figure_5_3",
    ]
