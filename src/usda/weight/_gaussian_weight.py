# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 21:33:01 2023

@author: richie bao
"""
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def gaussian_weight(da,figsize=None):    
    data=da.flatten()
    d_std=np.std(data) 
    d_mean=np.mean(data)
    cdf=norm.cdf(data,loc=d_mean,scale=d_std)
    gw_=1-cdf
    gw=gw_.reshape(da.shape)
    
    if figsize is not None: 
        fig,ax=plt.subplots(figsize=figsize)
        ax.plot(np.sort(data), np.sort(gw_)[::-1], label='1-cdf')
        plt.show()        
        return gw
    else:        
        return gw
