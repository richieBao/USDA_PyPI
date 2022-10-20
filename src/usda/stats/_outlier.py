# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:52:20 2022

@author: richie bao
"""
import numpy as np

def is_outlier(data,threshold=3.5):
    '''
    function-判断异常值
        
    Params:
        data - 待分析的数据，列表或者一维数组；list/array
        threshold - 判断是否为异常值的边界条件, The default is 3.5；float
        
    Returns
        is_outlier_bool - 判断异常值后的布尔值列表；list(bool)
        data[~is_outlier_bool] - 移除异常值后的数值列表；list
    '''     
    MAD=np.median(abs(data-np.median(data)))
    modified_ZScore=0.6745*(data-np.median(data))/MAD
    is_outlier_bool=abs(modified_ZScore)>threshold    
    return is_outlier_bool, data[~is_outlier_bool]
