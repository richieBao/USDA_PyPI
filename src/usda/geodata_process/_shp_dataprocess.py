# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 07:15:51 2023

@author: richie bao
"""
import geopandas as gpd
from tqdm import tqdm

def shp2gdf(fn,boundary=None,encoding='utf-8'):    
    '''
    转换.shp地理信息数据为GeoDataFrame(geopandas)数据格式，可以配置投影

    Parameters
    ----------
    fn : string
        SHP文件路径.
    boundary : .shp, optional
        配置裁切边界. The default is None.
    encoding : string, optional
        配置编码. The default is 'utf-8'.

    Returns
    -------
    GeoDataFrame
        读取SHP格式文件为GeoDataFrame格式返回.

    '''
    tqdm.pandas()  
    
    shp_gdf=gpd.read_file(fn,encoding=encoding)    
    if boundary is not None:        
        shp_gdf['mask']=shp_gdf.geometry.progress_apply(lambda row:row.is_valid)
        shp_gdf=shp_gdf[shp_gdf['mask']==True]
        shp_clip_gdf=gpd.clip(shp_gdf.to_crs(boundary.crs),boundary)    
        return shp_clip_gdf
    else:
        return shp_gdf
