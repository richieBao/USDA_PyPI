# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 17:43:18 2022

@author: richie bao
"""
import pandas as pd
from benedict import benedict  
import csv
import os
import pathlib
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pyproj import CRS

def csv2df(poi_fn_csv):    
    '''
    function-转换CSV格式的POI数据为pandas的DataFrame
    
    Params:
        poi_fn_csv - 存储有POI数据的CSV格式文件路径        

    Returns:
        poi_df - DataFrame(pandas)        
    '''
    
    n=0
    with open(poi_fn_csv, newline='',encoding='utf-8') as csvfile:
        poi_reader=csv.reader(csvfile)
        poi_dict={}    
        poiExceptions_dict={}
        for row in poi_reader:    
            if row:
                try:
                    row_benedict=benedict(eval(row[0]))  # 用eval方法，将字符串字典"{}"转换为字典{}
                    flatten_dict=row_benedict.flatten(separator='_')  # 展平嵌套字典
                    poi_dict[n]=flatten_dict
                except:                    
                    print("incorrect format of data_row number:%s"%n)                    
                    poiExceptions_dict[n]=row
            n+=1
    poi_df=pd.concat([pd.DataFrame(poi_dict[d_k].values(),index=poi_dict[d_k].keys(),columns=[d_k]).T for d_k in poi_dict.keys()], sort=True,axis=0)
    print("_"*50)
    for col in poi_df.columns:
        try:
            poi_df[col]=pd.to_numeric(poi_df[col])
        except:
            print("%s data type is not converted..."%(col))
    print("_"*50)
    print(".csv to DataFrame completed!")
    return poi_df

def poi_csv2GeoDF_batch(poi_paths,fields_extraction,save_path):    
    '''
    funciton - CSV格式POI数据批量转换为GeoDataFrame格式数据，需要调用转换CSV格式的POI数据为pandas的DataFrame函数csv2df(poi_fn_csv)
    
    Params:
        poi_paths - 文件夹路径为键，值为包含该文件夹下所有文件名列表的字典；dict
        fields_extraction - 配置需要提取的字段；list(string)
        save_path - 存储数据格式及保存路径的字典；string
        
    Returns:
        poisInAll_gdf - 提取给定字段的POI数据；GeoDataFrame（GeoPandas）
    '''        
    poi_df_dic={}
    i=0
    for key in poi_paths:
        for val in poi_paths[key]:
            poi_csvPath=os.path.join(key,val)
            poi_df=csv2df(poi_csvPath)  
            # 注释掉csv2df() 函数内部的print("%s data type is not converted..."%(col))语句，以pass替代，减少提示内容，避免干扰
            print(val)
            poi_df_path=pathlib.Path(val)
            poi_df_dic[poi_df_path.stem]=poi_df
            i+=1
            
    poi_df_concat=pd.concat(poi_df_dic.values(),keys=poi_df_dic.keys(),sort=True)
    # print(poi_df_concat.loc[['poi_0_delicacy'],:])  # 提取index为 'poi_0_delicacy'的行，验证结果
    poi_fieldsExtraction=poi_df_concat.loc[:,fields_extraction]
    poi_geoDF=poi_fieldsExtraction.copy(deep=True)
    poi_geoDF['geometry']=poi_geoDF.apply(lambda row:Point(row.location_lng,row.location_lat),axis=1) 
    crs_4326=CRS('epsg:4326')  # 配置坐标系统，参考：https://spatialreference.org/        
    poisInAll_gdf=gpd.GeoDataFrame(poi_geoDF,crs=crs_4326)
    
    poisInAll_gdf.to_pickle(save_path['pkl'])
    poisInAll_gdf.to_file(save_path['geojson'],driver='GeoJSON',encoding='utf-8')
    
    poisInAll_gdf2shp=poisInAll_gdf.reset_index() # 不指定level参数，例如Level=0，会把多重索引中的所有索引转换为列
    poisInAll_gdf2shp.rename(columns={
        'location_lat':'lat', 'location_lng':'lng',
        'detail_info_tag':'tag','detail_info_overall_rating':'rating', 'detail_info_price':'price'},inplace=True)
    poisInAll_gdf2shp.to_file(save_path['shp'],encoding='utf-8')
        
    return poisInAll_gdf

def json2gdf(json_fn,numeric_columns=None,epsg=None):
    '''
    读取.geojson(json)文件为GeoDataFrame格式文件，选择配置投影

    Parameters
    ----------
    json_fn : string
        文件路径.
    epsg : int, optional
        坐标投影系统，epsg编号. The default is None.

    Returns
    -------
    gdf : GeoDataFrmae
        转换后的GeoDataFrame格式文件.

    '''    
    gdf=gpd.read_file(json_fn)
    if epsg:
        gdf.to_crs(epsg,inplace=True)   
    print("fields_{}".format(gdf.columns))    
    if numeric_columns:
        gdf=gdf.astype(numeric_columns)

    return gdf