# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 10:33:25 2022

@author: richie bao
"""
from osgeo import ogr # osgeo包含在GDAL库中
import osmium as osm
import pandas as pd
import shapely.wkb as wkblib

import geopandas as gpd
import os
import datetime

def shpPolygon2OsmosisTxt(shape_polygon_fp,osmosis_txt_fp):
    '''
    function - 转换shape的polygon为osmium的polygon数据格式（.txt），用于.osm地图数据的裁切
    
    Params:
        shape_polygon_fp - 输入shape地理数据格式的polygon文件路径；string
        osmosis_txt_fp - 输出为osmosis格式的polygon数据格式.txt文件路径；string
        
    Returns:
        None
    '''    
    driver=ogr.GetDriverByName('ESRI Shapefile') # GDAL能够处理众多地理数据格式，此时调入了ESRI Shapefile数据格式驱动
    infile=driver.Open(shape_polygon_fp) # 打开.shp文件
    layer=infile.GetLayer() # 读取层
    f=open(osmosis_txt_fp,"w") 
    f.write("osmosis polygon\nfirst_area\n")
    
    for feature in layer: 
        feature_shape_polygon=feature.GetGeometryRef() 
        print(feature_shape_polygon) # 为polygon
        firsts_area_linearring=feature_shape_polygon.GetGeometryRef(0) # polygon不包含嵌套，为单独的形状
        print(firsts_area_linearring) # 为linearRing
        area_vertices=firsts_area_linearring.GetPointCount() # 提取linearRing对象的点数量
        for vertex in range(area_vertices): # 循环点，并向文件中写入点坐标
            lon, lat, z=firsts_area_linearring.GetPoint(vertex)  
            f.write("%s  %s\n"%(lon,lat))
    f.write("END\nEND")  
    f.close()  

class osmHandler(osm.SimpleHandler):    
    '''
    class-通过继承osmium类 class osmium.SimpleHandler读取.osm数据. 
    
    代码示例：
    osm_Chicago_fp=r"F:\data\osm_clip.osm" 
    osm_handler=osmHandler() 
    osm_handler.apply_file(osm_Chicago_fp,locations=True)     
    '''
    
    def __init__(self):
        osm.SimpleHandler.__init__(self)
        self.osm_node=[]
        self.osm_way=[]
        self.osm_area=[]
        
    def node(self,n):
        wkbfab=osm.geom.WKBFactory()
        
        wkb=wkbfab.create_point(n)
        point=wkblib.loads(wkb,hex=True)
        self.osm_node.append([
            'node',
            point,
            n.id,
            n.version,
            n.visible,
            pd.Timestamp(n.timestamp),
            n.uid,
            n.user,
            n.changeset,
            len(n.tags),
            {tag.k:tag.v for tag in n.tags},
            ])

    def way(self,w):    
        wkbfab=osm.geom.WKBFactory()
        
        try:
            wkb=wkbfab.create_linestring(w)
            linestring=wkblib.loads(wkb, hex=True)
            self.osm_way.append([
                'way',
                linestring,
                w.id,
                w.version,
                w.visible,
                pd.Timestamp(w.timestamp),
                w.uid,
                w.user,
                w.changeset,
                len(w.tags),
                {tag.k:tag.v for tag in w.tags}, 
                ])
        except:
            pass
        
    def area(self,a):  
        wkbfab=osm.geom.WKBFactory()        
        
        try:
            wkb=wkbfab.create_multipolygon(a)
            multipolygon=wkblib.loads(wkb, hex=True)
            self.osm_area.append([
                'area',
                multipolygon,
                a.id,
                a.version,
                a.visible,
                pd.Timestamp(a.timestamp),
                a.uid,
                a.user,
                a.changeset,
                len(a.tags),
                {tag.k:tag.v for tag in a.tags}, 
                ])
        except:
            pass        
    
def save_osm(osm_handler,osm_type,save_path=r"./data/",fileType="GPKG"):    
    '''
    function - 根据条件逐个保存读取的osm数据（node, way and area）
    
    Params:
        osm_handler - osm返回的node,way和area数据，配套类osmHandler(osm.SimpleHandler)实现；Class
        osm_type - 要保存的osm元素类型，包括"node"，"way"和"area"；string
        save_path - 保存路径。The default is "./data/"；string
        fileType - 保存的数据类型，包括"shp", "GeoJSON", "GPKG"。The default is "GPKG"；string
    
    Returns:
        osm_node_gdf - OSM的node类；GeoDataFrame(GeoPandas)
        osm_way_gdf - OSM的way类；GeoDataFrame(GeoPandas)
        osm_area_gdf - OSM的area类；GeoDataFrame(GeoPandas)
    '''       
    a_T=datetime.datetime.now()
    print("start time:",a_T)  

    def duration(a_T):
        b_T=datetime.datetime.now()
        print("end time:",b_T)
        duration=(b_T-a_T).seconds/60
        print("Total time spend:%.2f minutes"%duration)
        
    def save_gdf(osm_node_gdf,fileType,osm_type):
        if fileType=="GeoJSON":
            osm_node_gdf.to_file(os.path.join(save_path,"osm_%s.geojson"%osm_type),driver='GeoJSON')
        elif fileType=="GPKG":
            osm_node_gdf.to_file(os.path.join(save_path,"osm_%s.gpkg"%osm_type),driver='GPKG')
        elif fileType=="shp":
            osm_node_gdf.to_file(os.path.join(save_path,"osm_%s.shp"%osm_type))

    epsg_wgs84=4326 # 配置坐标系统，参考：https://spatialreference.org/        
    osm_columns=['type','geometry','id','version','visible','ts','uid','user','changeet','tagLen','tags']
    if osm_type=="node":
        osm_node_gdf=gpd.GeoDataFrame(osm_handler.osm_node,columns=osm_columns,crs=epsg_wgs84)
        save_gdf(osm_node_gdf,fileType,osm_type)
        duration(a_T)
        return osm_node_gdf

    elif osm_type=="way":
        osm_way_gdf=gpd.GeoDataFrame(osm_handler.osm_way,columns=osm_columns,crs=epsg_wgs84)
        save_gdf(osm_way_gdf,fileType,osm_type)
        duration(a_T)
        return osm_way_gdf
        
    elif osm_type=="area":
        osm_area_gdf=gpd.GeoDataFrame(osm_handler.osm_area,columns=osm_columns,crs=epsg_wgs84)
        save_gdf(osm_area_gdf,fileType,osm_type)
        duration(a_T)
        return osm_area_gdf
    
    
    
    
    
    
    
    
    
    
    
    