# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 20:41:10 2023

@author: richie bao
"""
import glob,os    
import pickle
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
import pyproj

  
def metrics_seg_pano_proportion(label_seg_path,img_Seg_path,coords):
    '''
    给定语义分割图像，计算各个对象占图像的百分比

    Parameters
    ----------
    label_seg_path : string
        语义分割标签根目录.
    img_Seg_path : string
        语义分割图像根目录.
    coords : dict
        各个道路对应全景图的采集坐标点.

    Returns
    -------
    panorama_object_percent_gdf : GeoDataFrame
        包含各个对象占比.

    '''
    # panorama_object_num=pd.DataFrame(columns=["fn_stem","fn_key","fn_idx","geometry",]+list(range(28)))
    label_mapping={
        0:"pole",
        1:"slight",
        2:"bboard",
        3:"tlight",
        4:"car",
        5:"truck",
        6:"bicycle",
        7:"motor",
        8:"bus",
        9:"tsignf",
        10:"tsignb",
        11:"road",
        12:"sidewalk",
        13:"curbcut",
        14:"crosspln",
        15:"bikelane",
        16:"curb",
        17:"fence",
        18:"wall",
        19:"building",
        20:"person",
        21:"rider",
        22:"sky",
        23:"vege",
        24:"terrain",
        25:"markings",
        26:"crosszeb",
        27:"Nan",                           
        }    
    label_seg_fns=glob.glob(os.path.join(label_seg_path,'*.pkl'))
    # print(label_seg_fns)
    panorama_object_num_lst=[]
    i=0
    for label_seg_fn in tqdm(label_seg_fns):
        with open(label_seg_fn,'rb') as f:
            label_seg=pickle.load(f)  
        fn_stem=Path(label_seg_fn).stem
        fn_key,fn_idx=fn_stem.split("_")      
        
        unique_elements, counts_elements=np.unique(label_seg, return_counts=True)
        object_frequency=dict(zip(unique_elements, counts_elements))
        object_frequency_update={k:object_frequency[k] if k in object_frequency.keys() else 0 for k in range(28) }
        coord=coords[fn_key][int(fn_idx)]
        object_frequency_update.update({"fn_stem":fn_stem,"fn_key":fn_key,"fn_idx":int(fn_idx),"geometry":Point(coord)})
        # panorama_object_num=panorama_object_num.append(object_frequency_update,ignore_index=True)
        panorama_object_num_lst.append(object_frequency_update)
        
        # if i==0:break
        # i+=1
    panorama_object_num=pd.DataFrame(panorama_object_num_lst,columns=["fn_stem","fn_key","fn_idx","geometry",]+list(range(28)))
    panorama_object_percent=panorama_object_num.copy(deep=True)
    panorama_object_percent[list(range(28))]=panorama_object_num[list(range(28))].div(np.prod(label_seg.shape)/100)   

    panorama_object_percent=panorama_object_percent.rename(columns=label_mapping)
    panorama_object_percent['ground_diff']=panorama_object_percent.apply(lambda row:100-row.sky-row.vege-row.building-row.wall-row.fence-row.bboard,axis=1)
    panorama_object_percent['sky_vege']=panorama_object_percent.apply(lambda row:row.sky+row.vege,axis=1)
    
    wgs84=pyproj.CRS('EPSG:4326')
    panorama_object_percent_gdf=gpd.GeoDataFrame(panorama_object_percent,geometry=panorama_object_percent.geometry,crs=wgs84) 

    return panorama_object_percent_gdf

def metrics_visual_entropy(panorama_object_percent_gdf):
    '''
    计算每幅语义分割图像，对象的信息和均衡度

    Parameters
    ----------
    panorama_object_percent_gdf : GeoDataFrame
        语义分割图像各个对象占图像的百分比.

    Returns
    -------
    GeoDataFrame
        包含对象的信息和均衡度.

    '''
    panorama_object_percent_gdf['ground']=panorama_object_percent_gdf.apply(lambda row:100-row.sky-row.vege-row.building,axis=1)
    def ve_row(row):
        import math
        import pandas as pd
        label=['pole', 'slight', 'bboard', 'tlight', 'car', 'truck', 'bicycle', 'motor', 'bus', 'tsignf', 'tsignb', 'road', 'sidewalk', 'curbcut', 'crosspln', 'bikelane', 'curb', 'fence', 'wall', 'building', 'person', 'rider', 'sky', 'vege', 'terrain', 'markings', 'crosszeb', 'Nan']
        ve=0.0
        for i in label:
            decimal_percentage=row[i]/100
            # print(decimal_percentage)
            if decimal_percentage!=0.:
                ve-=decimal_percentage*math.log(decimal_percentage)
        max_entropy=math.log(len(label))
        frank_e=ve/max_entropy*100
        
        return pd.Series([ve,frank_e])
    
    panorama_object_percent_gdf[['ve','equilibrium_degree']]=panorama_object_percent_gdf.apply(ve_row,axis=1)
    return panorama_object_percent_gdf

def percent_frequency(df,columns,bins,digits):
    '''
    计算给定区间的百分比频数（percent frequency）

    Parameters
    ----------
    df : DataFrame or GeoDataFrame
        DataFrame数据.
    columns : list
        需要计算的列字段列表.
    bins : list
        频数宽度列表.

    Returns
    -------
    frequency : DataFrame
        给定区间的百分比频数.

    '''
    import pandas as pd
    frequency=df[columns].apply(pd.Series.value_counts,bins=bins,)
    column_names=[]
    for column_name in columns:
        cn=column_name+'_percentage'
        frequency[cn]=round((frequency[column_name] / frequency[column_name].sum()) * 100,3)
        column_names.append(cn)
    return frequency,column_names

if __name__=="__main__":
    import usda.utils as usda_utils
    __C=usda_utils.AttrDict()
    args=__C
    __C.pano_path=r'G:\data\pano_dongxistreet\images_valid'
    __C.label_seg_path=r'G:\data\pano_dongxistreet\pano_seg\seg_label'   
    __C.face_size=1000
    __C.equi2cub_dir=r'G:\data\pano_dongxistreet\pano_projection_transforms'
    
    __C.pano_redefined_path=r'G:\data\pano_dongxistreet\pano_projection_transforms\img_seg_redefined_color'
    __C.output_shape=(1024,1024)
    __C.little_planet='little_planet_1'
    __C.polar_seg_img_dir=r'G:\data\pano_dongxistreet\polar_seg_img'
    
    __C.coords_street_fn=r'G:\data\pano_dongxistreet\coords_street.pickle'
    
    with open(args.coords_street_fn,'rb') as f: 
        coords=pickle.load(f)          
    
    #A.街道空间对象视域占比
    label_seg_path=os.path.join(args.equi2cub_dir,'cube_label_seg') #r'./processed data/label_seg_cube'    
    img_Seg_path=os.path.join(args.equi2cub_dir,'cube_img_seg') #r'./processed data/img_seg_cube'    
    cube_object_percent_gdf=metrics_seg_pano_proportion(label_seg_path,img_Seg_path,coords,)
    
        
