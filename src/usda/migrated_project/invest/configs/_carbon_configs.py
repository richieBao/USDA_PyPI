# -*- coding: utf-8 -*-
"""
Created on Sun May 21 15:18:37 2023

@author: richie bao
"""
import numpy as np

_OUTPUT_BASE_FILES = {
    'tot_c_cur': 'tot_c_cur.tif',
    'tot_c_fut': 'tot_c_fut.tif',
    'tot_c_redd': 'tot_c_redd.tif',
    'delta_cur_fut': 'delta_cur_fut.tif',
    'delta_cur_redd': 'delta_cur_redd.tif',
    'npv_fut': 'npv_fut.tif',
    'npv_redd': 'npv_redd.tif',
    'html_report': 'report.html',
}

_INTERMEDIATE_BASE_FILES = {
    'c_above_cur': 'c_above_cur.tif',
    'c_below_cur': 'c_below_cur.tif',
    'c_soil_cur': 'c_soil_cur.tif',
    'c_dead_cur': 'c_dead_cur.tif',
    'c_above_fut': 'c_above_fut.tif',
    'c_below_fut': 'c_below_fut.tif',
    'c_soil_fut': 'c_soil_fut.tif',
    'c_dead_fut': 'c_dead_fut.tif',
    'c_above_redd': 'c_above_redd.tif',
    'c_below_redd': 'c_below_redd.tif',
    'c_soil_redd': 'c_soil_redd.tif',
    'c_dead_redd': 'c_dead_redd.tif',
}

_TMP_BASE_FILES = {
    'aligned_lulc_cur_path': 'aligned_lulc_cur.tif',
    'aligned_lulc_fut_path': 'aligned_lulc_fut.tif',
    'aligned_lulc_redd_path': 'aligned_lulc_redd.tif',
}

# -1.0 since carbon stocks are 0 or greater
_CARBON_NODATA = -1.0
# use min float32 which is unlikely value to see in a NPV raster
_VALUE_NODATA = float(np.finfo(np.float32).min)




