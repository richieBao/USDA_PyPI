# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 16:15:05 2023

@author: richie bao
"""
from pyproj import Transformer,transform

import numpy as np
from shapely.geometry import MultiLineString
from shapely.ops import polygonize
import geopandas as gpd

def pt_coordi_transform(old_epsg,new_epsg,pt_coordinate):
    '''
    转换点坐标投影

    Parameters
    ----------
    old_epsg : pyproj.crs.crs.CRS
        点坐标投影.
    new_epsg : pyproj.crs.crs.CRS
        转换为的投影.
    pt_coordinate : list[float,float]
        点坐标.

    Returns
    -------
    list[float,float]
        转换投影后的点坐标.

    '''        

    transformer=Transformer.from_crs(old_epsg,new_epsg,always_xy=True)    
    return  transformer.transform(pt_coordinate[0], pt_coordinate[1])

def pt_on_quadrat(old_lb,old_rt,new_pt,position):  
    '''
    已知一个样方左下方法点坐标和右上方点坐标，及另一个样方一点坐标，求另一样方的对角点坐标

    Parameters
    ----------
    old_lb : list[float,float]
        已知样方左下角坐标.
    old_rt : list[float,float]
        已知样方右上角坐标.
    new_pt : list[float,float]
        待求样方点已知一点坐标.
    position : string
        待求样方点已知一点的位置，包括lb（左下角）,lt（左上角）,rb（右下角）,rt（右上角）.

    Returns
    -------
    list
        返回待求样方已知点的对角点.

    '''    
    lb_x,lb_y=old_lb
    rt_x,rt_y=old_rt
    
    width=rt_x-lb_x
    height=rt_y-lb_y
    
    pt_x,py_y=new_pt
    if position=='lb':
        return [pt_x+width,py_y+height]
    if position=='lt':
        return [pt_x+width,py_y-height]
    if position=='rb':
        return [pt_x-width,py_y+height]    
    if position=='rt':
        return [pt_x-width,py_y-height] 
    
def rec_quadrats_gdf(leftBottom_coordi,rightTop_coordi,h_distance,v_distance,crs=4326,to_crs=None):
    '''
    构建网格式样方

    Parameters
    ----------
    leftBottom_coordi : list(float)
        定位左下角坐标.
    rightTop_coordi : list(float)
        定位右上角坐标.
    h_distance : float
        单个样方宽度.
    v_distance : float
        单个样方长度.
    crs : int, optional
        投影编号. The default is 4326.
    to_crs : int, optional
        转换投影编号. The default is None.

    Returns
    -------
    grids_gdf : GeoDataFrame
        Polygon地理几何形式的GeoDataFrame格式样方数据.

    '''    
    
    x=np.arange(leftBottom_coordi[0], rightTop_coordi[0], h_distance)
    y=np.arange(leftBottom_coordi[1], rightTop_coordi[1], v_distance)
    hlines=[((x1, yi), (x2, yi)) for x1, x2 in zip(x[:-1], x[1:]) for yi in y]
    vlines=[((xi, y1), (xi, y2)) for y1, y2 in zip(y[:-1], y[1:]) for xi in x]
    grids=list(polygonize(MultiLineString(hlines + vlines)))
    
    grids_gdf=gpd.GeoDataFrame({'geometry':grids},crs=crs)
    if to_crs:
        grids_gdf.to_crs(to_crs,inplace=True)
        
    return grids_gdf    