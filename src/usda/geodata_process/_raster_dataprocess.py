# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 16:44:05 2023

@author: richie bao
"""
import rasterio as rio
from rasterio.windows import Window

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