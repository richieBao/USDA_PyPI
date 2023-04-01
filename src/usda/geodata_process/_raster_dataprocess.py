# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 16:44:05 2023

@author: richie bao
"""
import rasterio as rio
from rasterio.windows import Window
from osgeo import gdal, ogr, osr
from rasterio.warp import calculate_default_transform, reproject, Resampling

import rasterio,glob,os
from rasterio.merge import merge

def rio_read_subset(fn,lbNrt_coordinates):
    '''
    指定左下角和右上角坐标，部分读取栅格数据

    Parameters
    ----------
    fn : string
        栅格数据文件路径.
    lbNrt_coordinates :list[[float,float]] 
        左下角坐标和右上角坐标嵌套列表.

    Returns
    -------
    data : array
        根据左下角和右上角坐标提取的部分栅格数据.
    transform : affine.Affine
        投影变换.

    '''    
    
    lb_x,lb_y=lbNrt_coordinates[0]
    rt_x,rt_y=lbNrt_coordinates[1]        

    with rio.open(fn) as src:    
        lb_row_idx,lb_col_idx=src.index(lb_x, lb_y) # rio.transform.rowcol(src.transform,lb_x,lb_y)    
        rt_row_idx,rt_col_idx=src.index(rt_x,rt_y)
        height,width=lb_row_idx-rt_row_idx,rt_col_idx-lb_col_idx    
        window=Window(lb_col_idx,lb_row_idx-height,width,height)           
        data=src.read(window=window)         
        transform=rio.windows.transform(window,src.transform)
        ras_meta=src.profile
        
        ras_meta.update(
                width=width, 
                height=height,    
                transform=transform
                )          

    return data,transform,ras_meta

def raster2polygon(raster_in_path,shp_out_path,crs=None,band=1,dst_layer_name='values',field_name='values'):
    '''
    将栅格数据转换为Polygon对象的SHP格式矢量数据

    Parameters
    ----------
    raster_in_path : str
        待转换的栅格文件路径名.
    shp_out_path : str
        转换后SHP格式文件保存路径名.
    crs : int/str, optional
        投影epsg编号。如果不指定，则使用栅格自身的投影. The default is None.
    band : int, optional
        转换的波段. The default is 1.        
    dst_layer_name : str, optional
        层名称. The default is 'values'.
    field_name : str, optional
        字段名称，存储栅格数值. The default is 'values'.

    Returns
    -------
    None.

    '''
    src_ds=gdal.Open(raster_in_path)
    srcband=src_ds.GetRasterBand(band)
    dst_layername=dst_layer_name
    drv=ogr.GetDriverByName("ESRI Shapefile")
    dst_ds=drv.CreateDataSource(shp_out_path)
    prj=src_ds.GetProjection()
    
    sp_ref=osr.SpatialReference()
    if crs:        
        sp_ref.SetFromUserInput(crs)
    else:
        sp_ref.SetFromUserInput(prj)
        
    dst_layer=dst_ds.CreateLayer(dst_layername, srs=sp_ref)
    fld=ogr.FieldDefn(field_name, ogr.OFTInteger)
    dst_layer.CreateField(fld)
    dst_field=dst_layer.GetLayerDefn().GetFieldIndex(field_name)
    gdal.Polygonize(srcband, None, dst_layer, dst_field, [], callback=None) 

    del src_ds
    del dst_ds 
    
def raster_reprojection(raster_fp,save_path,dst_crs):

    '''
    function - 转换栅格投影
    
    Paras:
        raster_fp - 待转换投影的栅格
        dst_crs - 目标投影
        save_path - 保存路径
    '''
    with rio.open(raster_fp) as src:
        transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height,
            "compress":'lzw',
        })
        with rio.open(save_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)         

def raster_mosaic(dir_path,out_fp,dtype=None):
    '''
    function - 合并多个栅格为一个
    
    Paras:
        dir_path - 栅格根目录
        out-fp - 保存路径
    
    return:
        out_trans - 返回变换信息
    '''
    
    #迁移rasterio提供的定义数组最小数据类型的函数
    def get_minimum_int_dtype(values):
        """
        Uses range checking to determine the minimum integer data type required
        to represent values.

        :param values: numpy array
        :return: named data type that can be later used to create a numpy dtype
        """

        min_value = values.min()
        max_value = values.max()

        if min_value >= 0:
            if max_value <= 255:
                return rasterio.uint8
            elif max_value <= 65535:
                return rasterio.uint16
            elif max_value <= 4294967295:
                return rasterio.uint32
        elif min_value >= -32768 and max_value <= 32767:
            return rasterio.int16
        elif min_value >= -2147483648 and max_value <= 2147483647:
            return rasterio.int32
    
    search_criteria = "*.tif" #搜寻所要合并的栅格.tif文件
    fp_pattern=os.path.join(dir_path, search_criteria)
    fps=glob.glob(fp_pattern) #使用glob库搜索指定模式的文件
    src_files_to_mosaic=[]
    for fp in fps:
        src=rasterio.open(fp)
        src_files_to_mosaic.append(src)    
    mosaic,out_trans=merge(src_files_to_mosaic)  #merge函数返回一个栅格数组，以及转换信息   
    
    #获得元数据
    out_meta=src.meta.copy()
    #更新元数据
    if dtype:
        data_type=dtype
    else:
        data_type=get_minimum_int_dtype(mosaic)
    out_meta.update({"driver": "GTiff",
                     "height": mosaic.shape[1],
                     "width": mosaic.shape[2],
                     "transform": out_trans,
                     #通过压缩和配置存储类型，减小存储文件大小
                     "compress":'lzw',
                     "dtype":data_type, 
                      }
                    )      
    
    with rasterio.open(out_fp, "w", **out_meta) as dest:
        dest.write(mosaic.astype(data_type))     
    
    return out_trans