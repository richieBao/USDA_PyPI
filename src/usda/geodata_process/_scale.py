# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 09:27:57 2023

@author: richie bao
"""
from rasterio.enums import Resampling
import xarray
import numpy as np

def array2dataarray(data,transform,crs):
    shape=data.shape
    x=np.arange(0,shape[-1])*transform[0]+transform[2]
    y=np.arange(0,shape[-2])*transform[4]+transform[5]
    
    #x=np.arange(transform[2],transform[2]+(shape[-2]-1)*transform[0],transform[0])
    #y=np.arange(transform[5],transform[5]+(shape[-1]-1)*transform[4],transform[4])
    xda = xarray.DataArray(data=data,dims=['band','y','x'],coords=dict(x=x,y=y))
    xda.rio.write_transform(transform,inplace=True)
    xda.rio.write_crs(crs, inplace=True)

    return xda

def upNdownsampling(dataarray,scale_factor,resampling_method='mode'):
    new_width=dataarray.rio.width * scale_factor
    new_height=dataarray.rio.height * scale_factor    
    sampled=dataarray.rio.reproject(dataarray.rio.crs, shape=(int(new_height), int(new_width)), resampling=Resampling[resampling_method])
    return sampled

