# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 19:14:55 2023

@author: richie bao
"""
from src.util import general_util
import numpy as np

__C=general_util.AttrDict() 
gvals=__C
gvals.array=general_util.AttrDict() 
gvals.array.test=np.array([[1,2],[3,4]])