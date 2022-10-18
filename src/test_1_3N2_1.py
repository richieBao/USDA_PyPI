# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 08:49:18 2022

@author: richie bao
"""
import usda
from usda import *
from pathlib import Path
# print(dir(datasets))

def test_load_sales_data_cartoon_database():
    data=datasets.load_sales_data_cartoon_database()
    print(data)
    sales_table=data.sales_table
    print(sales_table)

def test_df2SQLite():
    data=datasets.load_sales_data_cartoon_database()
    sales_table=data.sales_table    
    db_fp="./test/database/fruits.sqlite"
    database.df2SQLite(db_fp,sales_table,"sales")  
    print(help(database.df2SQLite))
    
def test_SQLite2df():
    db_fp="./test/database/fruits.sqlite"
    data=database.SQLite2df(db_fp,"sales")
    print(data)

def test_postSQL2gpd():
    AoT_nodes_gdf=database.postSQL2gpd(table_name='AoT_nodes',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='AoT')
    print(AoT_nodes_gdf)
    
def test_DisplayablePath():  
    app_root=r'C:\Users\richi\Pictures'    
    paths = utils.DisplayablePath.make_tree(Path(app_root))
    for path in paths:
        print(path.displayable())    
        
def test_coordinate_transformation():
    lng = 128.543
    lat = 37.065
    # datasets._baiduPOI_dataCrawler.wgs84togcj02(lng, lat)
    # print(help(bdc))
    from usda.utils import _coordinate_transformation as cc
    print(help(cc))
    
def test_baiduPOI_dataCrawler_circle():
    page_num_range=range(20)
    query_dic={
        'location':'34.265708,108.953431',
        'radius':1000,
        'query':'旅游景点',   
        'page_size':'20',
        'scope':2, 
        'output':'json',
        'ak':'YuN8HxzYhGNfNLGX0FVo3NU3NOrgSNdF'        
    }    
    poi_save_path='./test/data/poi_circle.json'    
    datasets.baiduPOI_dataCrawler_circle(query_dic,poi_save_path,page_num_range)
    
def test_csv2df():
    poi_fn_csv='./test/data/poi_csv.csv'
    data=database.csv2df(poi_fn_csv)
    print(data)
    
def test_filePath_extraction():
    dirpath='./test'
    fileType=["csv","sqlite"]
    poi_paths=utils.filePath_extraction(dirpath,fileType)
    print(poi_paths) 
    
def test_poi_csv2GeoDF_batch():
    pass
    

if __name__=="__main__":
    print(dir(usda))
    print("-"*50)
    # test_load_sales_data_cartoon_database()
    # test_df2SQLite()
    # test_SQLite2df()
    # test_postSQL2gpd()
    # test_DisplayablePath()    
    # test_coordinate_transformation()
    # test_baiduPOI_dataCrawler_circle()
    # test_csv2df()
    # test_filePath_extraction()