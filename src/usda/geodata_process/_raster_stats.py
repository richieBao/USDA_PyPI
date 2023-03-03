# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 13:18:35 2023

@author: richi
"""
import rasterio as rio
import rasterstats as rst
import pandas as pd

def zonal_stats_raster(raster_fn,sampling_zone,band=1,stats=['majority'],add_stats=['frequency'],nodata=-9999):#
    '''
    区域统计，包括['count', 'min', 'max', 'mean', 'sum', 'std', 'median', 'majority', 'minority', 'unique', 'range', 'nodata', 'nan']，以及自定义的'frequency'，即频数统计

    Parameters
    ----------
    raster_fn : String
        待区域统计的栅格数据路径名.
    sampling_zone : GeoDataFrame
        用于栅格区域统计的polygon几何对象.
    band : int, optional
        数据波段. The default is 1.
    stats : List(String), optional
        默认统计的统计量名称. The default is ['majority'].
    add_stats :List(String) , optional
        自定义统计量名. The default is ['frequency'].

    Returns
    -------
    GeoDataFrame
        返回统计量值.

    '''
     
    sampling_zone_copy=sampling_zone.copy(deep=True)
    
    def frequency(x):
        data=x.data[~x.mask]
        return pd.value_counts(data)
    
    add_stats_dict={'frequency':frequency}
    with rio.open(raster_fn,'r') as src:
        band=src.read(band)
        sampling_zone_copy=sampling_zone_copy.to_crs(src.crs)
        zs_result=rst.zonal_stats(sampling_zone_copy,band,nodata=nodata,affine=src.transform,stats=stats,add_stats={i:add_stats_dict[i] for i in add_stats})
    
    for stat in stats:
        sampling_zone_copy[stat]=[dic[stat] for dic in zs_result]
    for stat  in add_stats:
        if stat=='frequency':
            fre=pd.concat([dic[stat].to_frame().T for dic in zs_result])
            fre.rename(columns={col:"{}_{}".format(stat,col) for col in fre.columns},inplace=True)
            fre.reset_index(inplace=True)  
    try:        
        zonal_stats_gdf=pd.concat([sampling_zone_copy,fre],axis=1)   
        
    except:
        zonal_stats_gdf=sampling_zone_copy
    return zonal_stats_gdf