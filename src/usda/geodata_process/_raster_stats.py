# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 13:18:35 2023

@author: richi
"""
import rasterio as rio
import rasterstats as rst
import pandas as pd
from rasterio.enums import Resampling
from shapely import Point
import math

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

def zonal_stats_raster_batch(raster_info,sampling_zone,band_n=1,nodata=-9999,upscale_mu=0):  
    '''
    多栅格文件批量区域统计，包括['count', 'min', 'max', 'mean', 'sum', 'std', 'median', 'majority', 'minority', 'unique', 'range', 'nodata', 'nan']，以及自定义的'frequency'，即频数统计

    Parameters
    ----------
    raster_info : dict[str:[str]]
        栅格文件路径名，统计方式字典，例如
        raster_info={
                'stories':[args.data.stories_fn,'mean'],
                'landuse':[args.data.landuse_fn,'frequency'],
                'landcover':[args.data.landcover_fn,'frequency'],
                'lst':[args.data.lst_fn,'mean'],
                'dem':[args.data.dem_fn,'mean'],
                'nightlight':[args.data.nightlight_fn,['mean']]
                }.
    sampling_zone : GeoDataFrame
        用于栅格区域统计的polygon几何对象.
    band_n : int, optional
        数据波段. The default is 1.
    nodata : int/float, optional
        空值. The default is -9999.

    Returns
    -------
    GeoDataFrame
        返回统计量值.

    '''

    def frequency(x):
        data=x.data[~x.mask]
        return pd.value_counts(data)    
    
    add_stats_dict={'frequency':frequency}    
    num=len(raster_info.keys())
    j=1
    zs_gdf=sampling_zone.copy(deep=True).reset_index(names=['on'])
    for k,v in raster_info.items():
        print(f'Processing img: {j}/{num}-{k}')
        sampling_zone_copy=sampling_zone.copy(deep=True).reset_index()
        with rio.open(v[0],'r') as src:
            sampling_zone_copy.to_crs(src.crs,inplace=True)
            
            box=sampling_zone_copy.geometry[0]
            b_x, b_y=box.exterior.coords.xy
            edge_length=(Point(b_x[0], b_y[0]).distance(Point(b_x[1], b_y[1])), Point(b_x[1], b_y[1]).distance(Point(b_x[2], b_y[2])))
            length=min(edge_length)
            cellsize=max(src.res)       
                
            if type(v[1]) is not list:
                v_1=[v[1]]
            else:
                v_1=v[1]

            stats=[]
            add_stats=[]        
            for i in v_1:
                if i in ['count', 'min', 'max', 'mean', 'sum', 'std', 'median', 'majority', 'minority', 'unique', 'range', 'nodata', 'nan']:
                    stats.append(i)
                elif i in list(add_stats_dict.keys()):
                    add_stats.append(i)
            if len(stats)==0:
                stats.append('count')
                
            if cellsize>length:
                upscale_factor=math.ceil(cellsize/length)+upscale_mu  
                print(f'resampling upscale={upscale_factor}')
                band=src.read(
                    out_shape=(
                        src.count,
                        int(src.height * upscale_factor),
                        int(src.width * upscale_factor)
                    ),
                    resampling=Resampling.bilinear
                    )
                
                transform=src.transform * src.transform.scale(
                    (src.width / band.shape[-1]),
                    (src.height / band.shape[-2])
                    )

                zs_result=rst.zonal_stats(sampling_zone_copy,band[0],nodata=nodata,affine=transform,stats=stats,add_stats={i:add_stats_dict[i] for i in add_stats})  
            else:
                band=src.read(band_n)
                zs_result=rst.zonal_stats(sampling_zone_copy,band,nodata=nodata,affine=src.transform,stats=stats,add_stats={i:add_stats_dict[i] for i in add_stats})  

            for stat in stats:
                zs_gdf[f'{k}_{stat}']=[dic[stat] for dic in zs_result]

            for stat in add_stats:
                if stat=='frequency':
                    fre=pd.concat([dic[stat].to_frame().T for dic in zs_result])
                    fre.rename(columns={col:"{}_{}_{}".format(k,stat[:3],col) for col in fre.columns},inplace=True)
                    fre.reset_index(drop=True,inplace=True)
                    fre.reset_index(inplace=True,names=['on'])    
                    zs_gdf=zs_gdf.merge(fre,on='on') 
                    zs_gdf.drop(columns=[f'{k}_count','on'],inplace=True,axis=1)                    
                    zs_gdf.reset_index(names=['on'],inplace=True)        
            j+=1
 
    return  zs_gdf