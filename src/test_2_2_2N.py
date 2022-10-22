# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 10:44:32 2022

@author: richie bao 
"""
import usda
from usda import *

def test_shpPolygon2OsmosisTxt():
    shape_polygon_small_fp='./test/data/OSMBoundary_small/OSMBoundary_small.shp'
    osmosis_txt_small_fp='./test/data/OSMBoundary_small.txt'
    data_process.shpPolygon2OsmosisTxt(shape_polygon_small_fp,osmosis_txt_small_fp)
    
def test_osmHandler():
    osm_Chicago_fp=r"./test\data\osm_small_clip.osm" 
    osm_handler=data_process.osmHandler()
    osm_handler.apply_file(osm_Chicago_fp,locations=True) 
    print(osm_handler)
    
def test_save_osm():
    osm_Chicago_fp=r"./test\data\osm_small_clip.osm" 
    osm_handler=data_process.osmHandler()
    osm_handler.apply_file(osm_Chicago_fp,locations=True) 
    node_gdf=data_process.save_osm(osm_handler,osm_type="area",save_path="./test/data",fileType="GPKG")
    print(node_gdf)
    
def test_duration():
    class A(object): 
        def __init__(self,x):
            self.x=x
        
    start_time=utils.start_time()
    f=[A(523825) for i in range(1000000)]
    utils.duration(start_time)
    
def test_pts2raster():
    amenity_kde_fn='./test/data/amenity_kde/amenity_kde.shp'
    raster_path='./test/data/amenity_epsg32616.tif'
    cellSize=300
    field_name='amenityKDE'
    poiRaster_array=data_process.pts2raster(amenity_kde_fn,raster_path,cellSize,field_name)
    print("conversion completed!")    
    print(poiRaster_array)
    
def test_ptsKDE_geoDF2raster():
    import geopandas as gpd
    
    fn=r"./test/data/amenity_kde/amenity_kde.shp" 
    amenity_poi=gpd.read_file(fn)
    raster_path_gpd='./test/data/amenity_kde_gdf.tif'
    cellSize=500 # cellSize值越小，需要计算的时间越长，开始调试时，可以尝试将其调大以节约计算时间
    scale=10**10 # 相当于math.pow(10,10)
    poiRasterGeoDF_array=stats.ptsKDE_geoDF2raster(amenity_poi,raster_path_gpd,cellSize,scale)  
    print(poiRasterGeoDF_array)
        
        
if __name__=="__main__":
    print(dir(usda))    
    print("-"*50)
    # test_shpPolygon2OsmosisTxt()
    # test_osmHandler()
    # test_save_osm()
    # test_duration()
    # test_pts2raster()
    test_ptsKDE_geoDF2raster()
    
    
    

