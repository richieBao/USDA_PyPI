# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 14:19:35 2022

@author: richie bao
"""
import math    

def deg2num(lat_deg, lon_deg, zoom):
    '''
    code migration
    function - 将经纬度坐标转换为指定zoom level缩放级别下，金子塔中瓦片的坐标。
    
    Params:
        lat_deg - 纬度；float
        lon_deg - 经度；float
        zoom - 缩放级别；int
        
    Returns:
        xtile - 金子塔瓦片x坐标；int
        ytile - 金子塔瓦片y坐标；int
    '''    
    
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    
    return (xtile, ytile)

def centroid(bounds):
    '''
    code migration
    function - 根据获取的地图边界坐标[左下角经度，左下角纬度，右上角经度，右上角维度]计算中心点坐标
    
    Params:
        bounds - [左下角经度，左下角纬度，右上角经度，右上角维度]；numerical
        
    Returns:
        lat - 边界中心点维度；float
        lng - 边界中心点经度；float
    '''
    
    lat=(bounds[1] + bounds[3]) / 2
    lng=(bounds[0] + bounds[2]) / 2
    
    return lat, lng
