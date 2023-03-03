# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 16:44:05 2023

@author: richie bao
"""
import rasterio as rio
from rasterio.windows import Window
from osgeo import gdal, ogr, osr
from rasterio.warp import calculate_default_transform, reproject, Resampling

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

