# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 08:02:39 2023

@author: richie bao
"""
from tqdm import tqdm
import glob,os 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import rasterio as rio
from rasterio.transform import from_origin
from pathlib import Path 
import pylandstats as pls
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
import pyproj  
import seaborn as sns


def metrics_skyline_shape(img_root,coords,hsv_lower,hsv_upper,save_root):
    '''
    天际线景观指数计算

    Parameters
    ----------
    img_root : string
        极坐标格式全景图所在根目录.
    coords : dict
        各个道路对应全景图的采集坐标点.
    hsv_lower : list
        颜色最小值控制.
    hsv_upper : list
        颜色最大值控制.
    save_root : string
        TIFF 格式图像保存根目录.

    Returns
    -------
    metrics_skyline_shape_gdf : GeoDataFrame
        天际线景观指数.

    '''     
    
    polar_seg_fns=glob.glob(os.path.join(img_root,'*.jpg'))
    hsv_lower_=np.asarray(hsv_lower)
    hsv_upper_=np.asarray(hsv_upper)
    
    transform=from_origin(472137, 5015782, 100, 100)  #472137, 5015782, 0.5, 0.5
    
    '''
    columns=["fn_stem","fn_key","fn_idx","geometry",]+['total_area', 'proportion_of_landscape', 'number_of_patches',
       'patch_density', 'largest_patch_index', 'total_edge', 'edge_density',
       'landscape_shape_index', 'effective_mesh_size', 'area_mn', 'area_am',
       'area_md', 'area_ra', 'area_sd', 'area_cv', 'perimeter_mn',
       'perimeter_am', 'perimeter_md', 'perimeter_ra', 'perimeter_sd',
       'perimeter_cv', 'perimeter_area_ratio_mn', 'perimeter_area_ratio_am',
       'perimeter_area_ratio_md', 'perimeter_area_ratio_ra',
       'perimeter_area_ratio_sd', 'perimeter_area_ratio_cv', 'shape_index_mn',
       'shape_index_am', 'shape_index_md', 'shape_index_ra', 'shape_index_sd',
       'shape_index_cv', 'fractal_dimension_mn', 'fractal_dimension_am',
       'fractal_dimension_md', 'fractal_dimension_ra', 'fractal_dimension_sd',
       'fractal_dimension_cv', 'euclidean_nearest_neighbor_mn',
       'euclidean_nearest_neighbor_am', 'euclidean_nearest_neighbor_md',
       'euclidean_nearest_neighbor_ra', 'euclidean_nearest_neighbor_sd',
       'euclidean_nearest_neighbor_cv']
    '''
    metrics=['total_area','area_mn','perimeter_mn','perimeter_area_ratio_mn','number_of_patches','landscape_shape_index','shape_index_mn','fractal_dimension_mn',]
    columns=["fn_stem","fn_key","fn_idx","geometry",]+metrics
    
    # sky_class_level_metrics=pd.DataFrame(columns=columns)    
    sky_class_level_metrics_lst=[]
    i=0
    for fn in tqdm(polar_seg_fns):
        fn_stem=Path(fn).stem
        fn_key,fn_idx=fn_stem.split("_")    
        coord=coords[fn_key][int(fn_idx)]
        img=cv2.imread(fn)
        img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask=cv2.inRange(img_hsv, hsv_lower_, hsv_upper_)        
        mask=np.where(mask==255,1,mask) #.astype(np.float64)       
        tiff_fn=os.path.join(save_root,'{}.tif'.format(Path(fn).stem))
        dst=rio.open(tiff_fn, 'w', driver='GTiff',
                                  height=mask.shape[0], width=mask.shape[1],
                                  count=1, dtype=str(mask.dtype),#dtype=rio.uint8,
                                  crs='+proj=utm +zone=10 +ellps=GRS80 +datum=NAD83 +units=m +no_defs',
                                  transform=transform)        
        dst.nodata=0
        dst.write(mask,1)
        dst.close()
        
        ls=pls.Landscape(tiff_fn)
        try:      
            class_metrics_df=ls.compute_class_metrics_df(metrics=metrics) 
            class_metrics_dict=class_metrics_df.transpose().to_dict()[1]
        except:
            class_metrics_dict={k:0 for k in metrics}              
        
        class_metrics_dict.update({"fn_stem":fn_stem,"fn_key":fn_key,"fn_idx":int(fn_idx),"geometry":Point(coord)})
        # sky_class_level_metrics=sky_class_level_metrics.append(class_metrics_dict,ignore_index=True)
        sky_class_level_metrics_lst.append(class_metrics_dict)
        
        # if i==3:break
        # i+=1
    sky_class_level_metrics=pd.DataFrame(sky_class_level_metrics_lst,columns=columns)
    wgs84='EPSG:4326' #pyproj.CRS('EPSG:4326')
    metrics_skyline_shape_gdf=gpd.GeoDataFrame(sky_class_level_metrics,geometry=sky_class_level_metrics.geometry,crs=wgs84) 
    return metrics_skyline_shape_gdf

def correlation_df(df,idxes,digits,save_path=None):
    '''
    有DataFrame数据格式计算相关系数

    Parameters
    ----------
    df : dataframe
        待计算的数据.
    idxes : list
        待计算的指数列名.
    save_path : string
        excel文件保存路径.
    digits : int
        保留小位数.

    Returns
    -------
    None.

    '''
    
    corr=df[idxes].corr()
    corr_round=corr.round(digits)
    if save_path:
        corr_round.to_excel(save_path)
    
    # Generate a mask for the upper triangle
    mask=np.triu(np.ones_like(corr, dtype=bool))
    
    # Set up the matplotlib figure
    f, ax=plt.subplots(figsize=(11, 9))    
    
    # Generate a custom diverging colormap
    cmap=sns.diverging_palette(230, 20, as_cmap=True)    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr_round, mask=mask, cmap=cmap, vmax=.3, center=0, annot=True, square=True, linewidths=.5, cbar_kws={"shrink": .5})    
   
    return corr_round