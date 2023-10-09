# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 19:51:11 2023

@author: richie bao
"""
from osgeo import gdal
from osgeo import gdal_array as gdn
from osgeo import ogr
import numpy as np
import tempfile 
import os

def rasterize(shp, attrib_name,cellSize=500,NoData_value=-9999,dtype='int32'):
    '''
    转换单个vector（polygon或points）shp格式数据为栅格数据

    Parameters
    ----------
    shp : String
        shp格式对象文件路径.
    attrib_name : String
        字段（属性）名.
    cellSize : numerical, optional
        栅格单元大小. The default is 500.
    dtype : String, optional
        数据类型，对应dtype_mapping = {'byte': gdal.GDT_Byte, 'uint8': gdal.GDT_Byte, 'uint16': gdal.GDT_UInt16, 'int8': gdal.GDT_Byte, 'int16': gdal.GDT_Int16, 'int32': gdal.GDT_Int32, 'uint32': gdal.GDT_UInt32, 'float32': gdal.GDT_Float32}. The default is 'int32'.

    Returns
    -------
    temp_out : TIFF
        栅格临时文件.

    '''
    
    # 定义空值（没有数据）的栅格数值 Define NoData value of new raster
    # NoData_value=-9999
    
    # 打开.shp点数据，并返回地理区域范围 Open the data source and read in the extent
    source_ds=ogr.Open(shp)
    source_layer=source_ds.GetLayer()
    x_min, x_max, y_min, y_max=source_layer.GetExtent()
    
    # 使用GDAL库建立栅格 Create the destination data source
    x_res=int((x_max - x_min) / cellSize)
    y_res=int((y_max - y_min) / cellSize)    
      
    # create empty raster (.tif) as temporary file and set its projection and extent to
    # that of the reference raster
    temp_out = tempfile.NamedTemporaryFile(suffix='.tif').name
    memory_driver = gdal.GetDriverByName('GTiff')
    dtype_mapping = {'byte': gdal.GDT_Byte, 'uint8': gdal.GDT_Byte, 'uint16': gdal.GDT_UInt16, 'int8': gdal.GDT_Byte, 'int16': gdal.GDT_Int16, 'int32': gdal.GDT_Int32, 'uint32': gdal.GDT_UInt32, 'float32': gdal.GDT_Float32}
    out_raster_ds = memory_driver.Create(temp_out, x_res, y_res, 1,dtype_mapping[dtype]) # gdal.GDT_Float64;gdal.GDT_Byte
    out_raster_ds.SetGeoTransform((x_min, cellSize, 0, y_max, 0, -cellSize))
    outband=out_raster_ds.GetRasterBand(1)
    outband.SetNoDataValue(NoData_value)

    # open shapefile vector layer to retrieve and burn attribute into the empty raster
    gdal.RasterizeLayer(out_raster_ds, [1], source_layer, options=["ATTRIBUTE="+attrib_name])
    return temp_out 

def img_to_array(input_file, dim_ordering="channel_last", dtype="float32"):
    '''
    将栅格文件各层栅格值转换为（numpy）数组

    Parameters
    ----------
    input_file : TIFF
        栅格文件.
    dim_ordering : String, optional
        调整波段位置. The default is "channel_last".
    dtype : Strng, optional
        栅格存储数据类型. The default is "float32".

    Returns
    -------
    arr : numpy.ndarray
        各层的栅格单元值.

    '''
    
    # open input raster, retrieve bands and convert to image array
    file = gdal.Open(input_file)
    bands = [file.GetRasterBand(i) for i in range(1, file.RasterCount +1)]
    arr = np.array([gdn.BandReadAsArray(band) for band in bands]).astype(dtype)
    
    # reoder dimensions so that channels/bands are last
    if dim_ordering=="channel_last": arr = np.transpose(arr, [1,2,0])
    return arr 

def array_to_tif(array,inVector,dst_filename,cellSize=500):
    '''
    将数组（含多个波段值）写入栅格文件

    Parameters
    ----------
    array : numpy.ndarray
        数组形式各层的栅格单元值.
    inVector : String
        shp格式对象文件路径.
    dst_filename : String
        待保存的栅格文件路径.
    cellSize : numerical, optional
        栅格单元大小. The default is 500.

    Returns
    -------
    None.

    '''

    # if os.path.exists(dst_filename):
    #     os.remove(dst_filename)
    #     print("The file has been deleted successfully")
    # else:
    #     print("The file does not exist!")    
    
    # NoData_value=-9999
    source_ds=ogr.Open(inVector)
    source_layer=source_ds.GetLayer()
    x_min, x_max, y_min, y_max=source_layer.GetExtent()
    x_res=int((x_max - x_min) / cellSize)
    y_res=int((y_max - y_min) / cellSize)      
    proj=source_layer.GetSpatialRef().ExportToWkt()
    
    # create empty raster (.tif) to which array will be written
    bands = array.shape[2]
    dtype = str(array.dtype)
    dtype_mapping = {'byte': gdal.GDT_Byte, 'uint8': gdal.GDT_Byte, 'uint16': gdal.GDT_UInt16, 'int8': gdal.GDT_Byte, 'int16': gdal.GDT_Int16, 'int32': gdal.GDT_Int32, 'uint32': gdal.GDT_UInt32, 'float32': gdal.GDT_Float32}
    driver = gdal.GetDriverByName('GTiff')
    output = driver.Create(dst_filename, x_res, y_res, bands, dtype_mapping[dtype])
    # set output image extent and projection
    output.SetGeoTransform((x_min, cellSize, 0, y_max, 0, -cellSize))
    output.SetProjection(proj)
    
    # write image array into empty raster
    for i in range(bands): output.GetRasterBand(i+1).WriteArray(array[:, :, i])
    output.FlushCache() 

def create_multiband_raster(attribs,inVector,dst_filename,cellSize=500,NoData_value=-9999,dtype='int32'):
    '''
    vector(SHP,.shp)格式文件转栅格，主程序
    ref_Rasterize: How to create multiband raster from vector attributes using python https://tkawuah.github.io/Blog1.html

    Parameters
    ----------
    attribs : List(String)
        待存储的.shp属性值列表.
    inVector : .shp
        .shp格式文件路径.
    dst_filename : Stirng-TIFF(.tiff)
        栅格保存路径名，通常以.tif为后缀名.
    cellSize : numercial, optional
        栅格单元大小. The default is 500.
    dtype : String, optional
        栅格单元存储数据的类型。dtype_mapping = {'byte': gdal.GDT_Byte, 'uint8': gdal.GDT_Byte, 'uint16': gdal.GDT_UInt16, 'int8': gdal.GDT_Byte, 'int16': gdal.GDT_Int16, 'int32': gdal.GDT_Int32, 'uint32': gdal.GDT_UInt32, 'float32': gdal.GDT_Float32}. The default is 'int32'.

    Returns
    -------
    None.

    '''
    
    img_array_list=[]
    for i in attribs:    
        fx=rasterize(inVector, i,cellSize=cellSize,dtype=dtype,NoData_value=NoData_value)
        fx_array=img_to_array(fx,dtype=dtype)
        img_array_list.append(fx_array)          

    fx_multi=np.concatenate(img_array_list, axis=-1)
    array_to_tif(fx_multi,inVector,dst_filename,cellSize=cellSize)  
    print('The raster was written successfully!')
