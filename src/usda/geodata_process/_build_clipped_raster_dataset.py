# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 14:22:09 2023

@author: richie bao
"""
from multiprocessing import Pool
import multiprocessing 
import geopandas as gpd
import numpy as np
from tqdm import tqdm
from sklearn.utils import Bunch
import pickle
import rioxarray as rxr
import pandas as pd
from functools import partial

import warnings
warnings.filterwarnings('ignore')

if __package__:
    from ._build_clipped_raster_dataset_pool import data_target_func
else:
    from _build_clipped_raster_dataset_pool import data_target_func

def xy_size_elevation(clipper_fn,raster_fn,n=1000):
    raster= rxr.open_rasterio(raster_fn)      
    clipper_gdf=gpd.read_file(clipper_fn)    
    clipper_gdf.to_crs(raster.rio.crs,inplace=True)    
    
    samples=clipper_gdf.sample(n=n)
    shape_lst=[]
    for idx,row in samples.iterrows():   
        try:
            clipped_raster=raster.rio.clip([row.geometry],from_disk=True)
            shape_lst.append(clipped_raster.shape)
        except:
            pass
    
    shape_df=pd.DataFrame(shape_lst)
    val_counts=shape_df.value_counts()
    print(val_counts)
    return val_counts

def build_clipped_raster_dataset(clipper_fn,raster_fn,dataset_save_fn,y_columns,x_size,y_size,x_offset=5,y_offset=10,ratio_cpu=0.5,ratio_split=1):    
    raster= rxr.open_rasterio(raster_fn)      
    clipper_gdf=gpd.read_file(clipper_fn)    
    clipper_gdf.to_crs(raster.rio.crs,inplace=True)
        
    cpus = multiprocessing.cpu_count()

    cpus_used=int(cpus*ratio_cpu)
    args=partial(data_target_func, args=[raster,y_columns,x_size,y_size,x_offset,y_offset])
    print(f'cpu used num={cpus_used};batch size={cpus_used*ratio_split}')
    
    with Pool(cpus_used) as p:
        data_target=p.map(args, tqdm(np.array_split(clipper_gdf,cpus_used*ratio_split)))
        # data_target=p.map(data_target_func,tqdm(np.array_split(LST_rank,cpus_used*ratio_split)))

    data_target=[x for x in data_target if x is not None]   
    data_lst,target_lst=zip(*data_target)
    data=np.vstack(data_lst)
    # target=np.hstack(target_lst)
    target=np.vstack(target_lst)
    # print(target[:,0])
    LC2LST_dataset=Bunch(data=data, target=target[:,0],extra=target[:,1:])
    
    with open(dataset_save_fn,'wb') as f:
        pickle.dump(LC2LST_dataset,f)  
        
    # return LC2LST_dataset    

if __name__=="__main__":    
    clipper_fn=r'I:\data\london\LST_rank.shp'  
    raster_fn=r'I:\\data\\ESA_London\\ESA_WorldCover_10m_2020_v100_N51W003_Map.tif'
    dataset_save_fn='I:\data\london\LC2LST_dataset.pickle'
    
    # shape_fre=xy_size_elevation(clipper_fn,raster_fn)    
    build_clipped_raster_dataset(clipper_fn,raster_fn,dataset_save_fn,['rank_Jenks','y'],90,150,5,10,ratio_cpu=0.7,ratio_split=30) 

    with open(dataset_save_fn,'rb') as f:
        ds=pickle.load(f)