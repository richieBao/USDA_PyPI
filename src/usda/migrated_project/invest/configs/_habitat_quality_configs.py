# -*- coding: utf-8 -*-
"""
Created on Mon May 22 02:23:08 2023

@author: richie bao
"""
import numpy

# All out rasters besides rarity should be gte to 0. Set nodata accordingly.
_OUT_NODATA = float(numpy.finfo(numpy.float32).min)
# Scaling parameter from User's Guide eq. 4 for quality of habitat
_SCALING_PARAM = 2.5
# To help track and name threat rasters from paths in threat table columns
_THREAT_SCENARIO_MAP = {'_c': 'cur_path', '_f': 'fut_path', '_b': 'base_path'}

