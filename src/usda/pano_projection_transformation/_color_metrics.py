# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 09:13:38 2023

@author: richie bao
"""
import glob
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
import pyproj
import geopandas as gpd
import pickle   
import math,copy
import pyproj
import os

if __package__:
    from ._color_metrics_pool import find_dominant_colors_pool
    from ._color_metrics_pool import dominant2cluster_colors_pool
else:
    from _color_metrics_pool import find_dominant_colors_pool
    from _color_metrics_pool import dominant2cluster_colors_pool    

def find_dominant_colors_pool_main(img_path,args_,cpu_num=8):
    '''
    计算图像的主题色(多进程)

    Parameters
    ----------
    img_path : string
        图像根目录.
    args_ : list
        包括[coords,resize_scale,number_of_colors].

    Returns
    -------
    img_dominant_color_gdf : ＧｅｏＤａｔａＦｒａｍｅ
        图像的主题色.

    '''
    
    coords,resize_scale,number_of_colors=args_
    img_fns=glob.glob(os.path.join(img_path,'*.jpg')) #[:3]
    args=partial(find_dominant_colors_pool, args=args_)
    with Pool(cpu_num) as p:
        color_dic_list=p.map(args, tqdm(img_fns))  

    # img_dominant_color=pd.DataFrame(columns=["fn_stem","fn_key","fn_idx","geometry",]+list(range(number_of_colors)))
    img_dominant_color_lst=[]
    for color_dic in color_dic_list:    
        # img_dominant_color=img_dominant_color.append(color_dic,ignore_index=True)
        img_dominant_color_lst.append(color_dic)
    
    img_dominant_color=pd.DataFrame(img_dominant_color_lst,columns=["fn_stem","fn_key","fn_idx","geometry",]+list(range(number_of_colors)))
    wgs84=pyproj.CRS('EPSG:4326')
    img_dominant_color_gdf=gpd.GeoDataFrame(img_dominant_color,geometry=img_dominant_color.geometry,crs=wgs84)
    return img_dominant_color_gdf

def dominant2cluster_colors_pool_main(img_path,colors_dominant2cluster_path,args_,cpu_num=8):
    '''
    主题色聚类(多进程)

    Parameters
    ----------
    img_path : string
        图像根目录.
    colors_dominant2cluster_path : string
        文件保持路径.
    args_ : list
        包括[coords,resize_scale,number_of_colors].

    Returns
    -------
    color_dic_list : dict
        主题色聚类信息.

    '''
    
    img_fns=glob.glob(os.path.join(img_path,'*.jpg')) #[:3]
    args=partial(dominant2cluster_colors_pool, args=args_)    
    with Pool(cpu_num) as p:
        color_dic_list=p.map(args, tqdm(img_fns))        
    with open(colors_dominant2cluster_path,'wb') as f: 
        pickle.dump(color_dic_list,f) 
        
    return color_dic_list
        
def colors_entropy(colors_dominant_clustering_fn):
    '''
    计算图像主题色聚类信息熵

    Parameters
    ----------
    colors_dominant_clustering_fn : string
        主题色聚类信息.

    Returns
    -------
    GeoDataFrame
        图像主题色聚类信息熵.

    '''    
    with open(colors_dominant_clustering_fn,'rb') as f:
        colors_dominant_clustering=pickle.load(f) #[:10]
    print(sum(colors_dominant_clustering[0]['counter'].values()))
    def entropy(counter_dict):
        percentage=[i/sum(counter_dict.values()) for i in counter_dict.values()]
        ve=0.0
        for perc in percentage:
            if perc!=0.:
                ve-=perc*math.log(perc)            
        max_entropy=math.log(len(counter_dict.keys()))
        frank_e=ve/max_entropy*100    
        return frank_e
            
    color_dominant_entropy=[entropy(dic['counter']) for dic in tqdm(colors_dominant_clustering)]
    wgs84=pyproj.CRS('EPSG:4326')
    colors_dominant_clustering_copy=copy.deepcopy(colors_dominant_clustering)
    [colors_dominant_clustering_copy[i].update({'counter':color_dominant_entropy[i]}) for i in range(len(colors_dominant_clustering_copy))]
    
    # colors_dominant_entropy=pd.DataFrame(columns=["fn_stem","fn_key","fn_idx","geometry",'counter'])
    colors_dominant_entropy_lst=[]
    for colors_dominant_dic in colors_dominant_clustering_copy:    
        # colors_dominant_entropy=colors_dominant_entropy.append(colors_dominant_dic,ignore_index=True)        
        colors_dominant_entropy_lst.append(colors_dominant_dic)
    
    colors_dominant_entropy=pd.DataFrame(colors_dominant_entropy_lst,columns=["fn_stem","fn_key","fn_idx","geometry",'counter'])
    colors_dominant_entropy_gdf=gpd.GeoDataFrame(colors_dominant_entropy,geometry=colors_dominant_entropy.geometry,crs=wgs84)        
    return colors_dominant_entropy_gdf

