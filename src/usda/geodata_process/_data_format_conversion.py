# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 08:15:58 2023

@author: richie bao
"""
import pandas as pd
import geopandas as gpd
import fiona,io
from tqdm import tqdm

def kml2gdf_folder(fn,epsg=None,boundary=None): 
    '''
    转换.kml（Google Eath）为GeoDataFrame格式

    Parameters
    ----------
    fn : .kml
        在Google Earth中绘制的格式.
    epsg : epsg编号, optional
        地理投影信息.
    boundary : GeoDataFrame, optional
        裁切边界. The default is None.

    Returns
    -------
    kml_gdf_proj : GeoDataFrame
        转换.kml为GeoDataFrame.

    '''
    # Enable fiona driver
    fiona.drvsupport.supported_drivers['KML'] = 'rw'
    
    kml_gdf=gpd.GeoDataFrame()
    for layer in tqdm(fiona.listlayers(fn)):
        src=fiona.open(fn, layer=layer)
        meta = src.meta
        meta['driver'] = 'KML'        
        with io.BytesIO() as buffer:
            with fiona.open(buffer, 'w', **meta) as dst:            
                for i, feature in enumerate(src):
                    if len(feature['geometry']['coordinates'][0]) > 1:                       
                        dst.write(feature)
            buffer.seek(0)
            one_layer=gpd.read_file(buffer,driver='KML')

            one_layer['group']=layer      
            kml_gdf=pd.concat([kml_gdf,one_layer])            

    if epsg is not None:
        kml_gdf.to_crs(epsg=epsg,inplace=True)
        
    if boundary:
        kml_gdf_proj['mask']=kml_gdf.geometry.apply(lambda row:row.within(boundary))
        kml_gdf_proj.query('mask',inplace=True)    
    else:
        kml_gdf_proj=kml_gdf
        
    return kml_gdf_proj 
