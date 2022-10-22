# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 10:38:00 2022

@author: richie bao
"""
from osgeo import gdal,ogr,osr
import numpy as np
from scipy import stats

def ptsKDE_geoDF2raster(pts_geoDF,raster_path,cellSize,scale):
    '''
    function - 计算GeoDaraFrame格式的点数据核密度估计，并转换为栅格数据
    
    Params:
        pts_geoDF - GeoDaraFrame格式的点数据；GeoDataFrame(GeoPandas)
        raster_path - 保存的栅格文件路径；string
        cellSize - 栅格单元大小；int
        scale - 缩放核密度估计值；int/float
        
    Returns:
        返回读取已经保存的核密度估计栅格数据；array
    '''    
    # 定义空值（没有数据）的栅格数值 Define NoData value of new raster
    NoData_value=-9999
    x_min, y_min,x_max, y_max=pts_geoDF.geometry.total_bounds

    # 使用GDAL库建立栅格 Create the destination data source
    x_res=int((x_max - x_min) / cellSize)
    y_res=int((y_max - y_min) / cellSize)
    target_ds=gdal.GetDriverByName('GTiff').Create(raster_path, x_res, y_res, 1, gdal.GDT_Float64 )
    target_ds.SetGeoTransform((x_min, cellSize, 0, y_max, 0, -cellSize))
    outband=target_ds.GetRasterBand(1)
    outband.SetNoDataValue(NoData_value)   
    
    # 配置投影坐标系统
    spatialRef = osr.SpatialReference()
    epsg=int(pts_geoDF.crs.srs.split(":")[-1])
    spatialRef.ImportFromEPSG(epsg)  
    target_ds.SetProjection(spatialRef.ExportToWkt())
    
    # 向栅格层中写入数据
    X, Y = np.meshgrid(np.linspace(x_min,x_max,x_res), np.linspace(y_min,y_max,y_res)) # 用于定义提取核密度估计值的栅格单元坐标数组
    positions=np.vstack([X.ravel(), Y.ravel()])
    values=np.vstack([pts_geoDF.geometry.x, pts_geoDF.geometry.y])    
    print("Start calculating kde...")
    kernel=stats.gaussian_kde(values)
    Z=np.reshape(kernel(positions).T, X.shape)
    print("Finish calculating kde!")
    # print(values)
        
    outband.WriteArray(np.flip(Z,0)*scale) # 需要翻转数组，写栅格单元        
    outband.FlushCache()
    print("conversion completed!")
    return gdal.Open(raster_path).ReadAsArray()

