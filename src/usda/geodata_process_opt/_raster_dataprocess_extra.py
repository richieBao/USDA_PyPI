# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 06:57:41 2023

@author: richie bao
"""
import rasterio as rio
from rasterio.enums import Resampling

def raster_resampling(raster_fn,output_fn,xres,yres):
    with rio.open(raster_fn) as dataset:
        scale_factor_x = dataset.res[0]/xres
        scale_factor_y = dataset.res[1]/yres
    
        profile = dataset.profile.copy()
        # resample data to target shape
        data = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height * scale_factor_y),
                int(dataset.width * scale_factor_x)
            ),
            resampling=Resampling.bilinear
        )
    
        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (1 / scale_factor_x),
            (1 / scale_factor_y)
        )
        profile.update({"height": data.shape[-2],
                        "width": data.shape[-1],
                       "transform": transform})
    
    with rio.open(output_fn, "w", **profile) as dataset:
        dataset.write(data) 
