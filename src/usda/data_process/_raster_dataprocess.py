# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 07:46:40 2022

@author: richie bao
"""
import earthpy.spatial as es
import geopandas as gpd
from pyproj import CRS
import rasterio as rio

def raster_clip(raster_fp,clip_boundary_fp,save_path):  
    '''
    function - 给定裁切边界，批量裁切栅格数据
    
    Params:
        raster_fp - 待裁切的栅格数据文件路径（.tif）；string
        clip_boundary - 用于裁切的边界（.shp，WGS84，无投影），与栅格具有相同的坐标投影系统；string
    
    Returns:
        rasterClipped_pathList - 裁切后的文件路径列表；list(string)
    '''
    
    clip_bound=gpd.read_file(clip_boundary_fp)
    with rio.open(raster_fp[0]) as raster_crs:
        raster_profile=raster_crs.profile
        clip_bound_proj=clip_bound.to_crs(raster_profile["crs"])
    
    rasterClipped_pathList=es.crop_all(raster_fp, save_path, clip_bound_proj, overwrite=True) # 对所有波段band执行裁切
    print("clipping finished.")
    return rasterClipped_pathList
