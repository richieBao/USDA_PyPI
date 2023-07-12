# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 17:30:43 2023

@author: richie bao
"""
import math  
# from rio_tiler.io import COGReader # proj problem
import numpy as np
import matplotlib.pyplot as plt

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

def tiled_web_map_show(fn,z=9,tilesize=512,offset_x=0,offset_y=0,cmap=None,centroid_latlon=None,figsize=(10,10)):
    '''
    以金子塔缩放（zoom）形式（tiled web map）显示地图（tif等影像文件）

    Parameters
    ----------
    fn : str
        影像文件路径.
    z : int, optional
        球面墨卡托投影金字塔缩放比例（0-23级）. The default is 9.
    tilesize : int, optional
        输出图像的大小. The default is 512.
    offset_x : int, optional
        显示影像位置x向偏移. The default is 0.
    offset_y : int, optional
        显示影像位置y向偏移. The default is 0.
    cmap : matplotlib.colors.ListedColormap, optional
        颜色配置. The default is None.
    figsize : tuple(int,int), optional
        打印图像大小. The default is (10,10).

    Returns
    -------
    img : array
    图像数据.

    '''
    with COGReader(fn) as src:
        print('CRS:',src.crs)
        print('影像边界坐标：',src.geographic_bounds)
        if centroid_latlon:
            x,y=deg2num(*centroid_latlon,z)
        else:
            x, y=deg2num(*centroid(src.geographic_bounds), z) # 指定缩放级别，转换影像中心点的经纬度坐标为金子塔瓦片坐标
        img=src.tile(x+offset_x,y+offset_y,z, tilesize=tilesize) # tilesize参数为瓦片大小，默认值为256
        
    tile=img.data       
    tile=np.transpose(tile, (1, 2, 0))
    print("瓦片的形状：",tile.shape)
       
    plt.figure(figsize=figsize)
    plt.imshow(tile,cmap=cmap)
    plt.axis("off")
    plt.show()
    
    return img
    