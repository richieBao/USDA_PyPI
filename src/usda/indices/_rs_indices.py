# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 14:08:30 2022

@author: richie bao
"""

def NDVI(RED_band,NIR_band):
    '''
    function - 计算NDVI指数
    
    Params:
        RED_band - 红色波段；array
        NIR_band - 近红外波段；array
        
    Returns:
        NDVI - NDVI指数值；array
    '''
    
    RED_band=np.ma.masked_where(NIR_band+RED_band==0,RED_band)
    NDVI=(NIR_band-RED_band)/(NIR_band+RED_band)
    NDVI=NDVI.filled(-9999)
    print("NDVI"+"_min:%f,max:%f"%(NDVI.min(),NDVI.max()))
    
    return NDVI