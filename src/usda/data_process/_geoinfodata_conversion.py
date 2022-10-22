# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 09:44:46 2022

@author: richie bao
"""
from osgeo import gdal,ogr

def pts2raster(pts_shp,raster_path,cellSize,field_name=False):       
    '''
    function - 将.shp格式的点数据转换为.tif栅格(raster)
               将点数据写入为raster数据。使用raster.SetGeoTransform，栅格化数据。参考GDAL官网代码 
    
    Params:
        pts_shp - .shp格式点数据文件路径；SHP点数据
        raster_path - 保存的栅格文件路径；string
        cellSize - 栅格单元大小；int
        field_name - 写入栅格的.shp点数据属性字段；string
        
    Returns:
        返回读取已经保存的栅格数据；array
    '''   
    # 定义空值（没有数据）的栅格数值 Define NoData value of new raster
    NoData_value=-9999
    
    # 打开.shp点数据，并返回地理区域范围 Open the data source and read in the extent
    source_ds=ogr.Open(pts_shp)
    source_layer=source_ds.GetLayer()
    x_min, x_max, y_min, y_max=source_layer.GetExtent()
    
    # 使用GDAL库建立栅格 Create the destination data source
    x_res=int((x_max - x_min) / cellSize)
    y_res=int((y_max - y_min) / cellSize)
    target_ds=gdal.GetDriverByName('GTiff').Create(raster_path, x_res, y_res, 1, gdal.GDT_Float64) #create(filename,x_size,y_size,band_count,data_type,creation_options)。gdal的数据类型 gdal.GDT_Float64,gdal.GDT_Int32...
    target_ds.SetGeoTransform((x_min, cellSize, 0, y_max, 0, -cellSize))
    outband=target_ds.GetRasterBand(1)
    outband.SetNoDataValue(NoData_value)

    # 向栅格层中写入数据
    if field_name:
        gdal.RasterizeLayer(target_ds,[1], source_layer,options=["ATTRIBUTE={0}".format(field_name)])
    else:
        gdal.RasterizeLayer(target_ds,[1], source_layer,burn_values=[-1])   
        
    # 配置投影坐标系统
    spatialRef=source_layer.GetSpatialRef()
    target_ds.SetProjection(spatialRef.ExportToWkt())       
        
    outband.FlushCache()
    return gdal.Open(raster_path).ReadAsArray()