# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 17:59:29 2023

@author: richie bao
"""
from shapely.geometry import Polygon
import shapely 
from shapely.geometry import shape
from tqdm import tqdm 
import geopandas as gpd

def IoU_2Polygons(polygon1,polygon2):
    '''
    计算两个Poygon（Shapely）对象的交并比

    Parameters
    ----------
    polygon1 : POLYGON（Shapely）
        多边形对象1.
    polygon2 : POLYGON（Shapely）
        多边形对象2.

    Returns
    -------
    iou : float
        交并比 Intersection over Union，IoU.

    '''       
    intersect_area=polygon1.intersection(polygon2).area
    union_area=polygon1.union(polygon2).area
    iou=intersect_area/union_area
    
    return iou

def drop_overlapping_polygons(gdf_,iou=0.5):    
    '''
    移除GeoDataFrame格式文件Polygon对象重叠的行。保留第一个出现的对象，而移除后面与之重叠的对象

    Parameters
    ----------
    gdf_ : GeoDataFrame
        为Polygon对象的地理信息数据.
    iou : float, optional
        交并比（Intersection over Union，IoU）. The default is 0.5.

    Returns
    -------
    gdf_non_overlapping : GeoDataFrame
        移除重叠的Polygon后的GeoDataFrame格式数据.

    '''    
    gdf=gdf_.copy(deep=True)
    polygons_dict=gdf['geometry'].to_dict()    
    tabu_idx=[]
    for idx,row in tqdm(gdf.iterrows(),total=gdf.shape[0]):  
        tabu_idx.append(idx)
        polygons_except4one_dict={key:value for key, value in polygons_dict.items() if key not in tabu_idx}
        pg_gdf=row.geometry
        for k,pg_dict in polygons_except4one_dict.items():  
            iou_2pgs=IoU_2Polygons(pg_dict,pg_gdf)    
            if iou_2pgs>iou:
                polygons_dict.pop(k)
                
    gdf_non_overlapping=gdf.loc[list(polygons_dict.keys())]         
    return gdf_non_overlapping

def planetary_computer_items_filter4download(items,border=None,resetIDX=True):
    d={'idx':[],'id':[],'datetime':[],'url':[],'geometry':[]}
    for idx,item in enumerate(items):
        d['idx'].append(idx)
        d['id'].append(item.id)
        d['datetime'].append(item.properties['datetime'])
        
        url_=item.assets['image'].href
        url=url_.split('?')[0]                
        d['url'].append(url)
        d['geometry'].append(shape(item.geometry))
 
    items_gdf=gpd.GeoDataFrame(d,crs="EPSG:4326")
    if border is not None:
        #print(border_polygon)
        def overlap_func(row):
            #print(row.geometry)
            within_tf=shapely.within(row.geometry,border)
            intersects=shapely.intersects(row.geometry,border)
            
            if within_tf==True or intersects==True :return 1
            else: return 0
        
        items_gdf['within']=items_gdf.apply(overlap_func,axis=1)
        items_gdf=items_gdf[items_gdf['within']==1]  
        items_gdf.drop(columns=['within'],inplace=True)
        
    if resetIDX:
        items_gdf.reset_index(drop=True, inplace=True)
    
    return items_gdf