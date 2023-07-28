# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 09:17:16 2023

@author: richie bao
"""
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 15:23:09 2021
Created on Tue Jan 25 16:41:36 2022

@author: Richie Bao-caDesign设计(cadesign.cn)
"""
import glob,os  
from tqdm import tqdm
import cv2
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from shapely.geometry import Point
import pyproj
import geopandas as gpd
import numpy as np
from skimage import measure

def RGB2HEX(color):
    '''
    转换RGB色彩位HEX格式

    Parameters
    ----------
    color : list
        RGB颜色值.

    Returns
    -------
    string
        HEX格式颜色值.

    '''
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))
def get_image(image_path):
    '''
    cv2方法读取图像

    Parameters
    ----------
    image_path : string
        图像路径.

    Returns
    -------
    image : array/list
        RGB颜色值.

    '''      
    image=cv2.imread(image_path)
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def find_dominant_colors(imgs_root,coords,resize_scale=0.5,number_of_colors=10,show_chart=False):
    '''
    计算图像的主题色

    Parameters
    ----------
    imgs_root : string
        图像根目录.
    coords : dict
        各个道路对应全景图的采集坐标点.
    resize_scale : numerical val, optional
        调整图像大小的比例. The default is 0.5.
    number_of_colors : int, optional
        主题色提取的数量. The default is 10.
    show_chart : bool, optional
        是否打印主题色的饼状图. The default is False.

    Returns
    -------
    img_dominant_color_gdf : GeoDataFrame
        图像的主题色.

    '''    
    img_fns=glob.glob(os.path.join(imgs_root,'*.jpg'))[:1] #[-133:]
    img_dominant_color=pd.DataFrame(columns=["fn_stem","fn_key","fn_idx","geometry",]+list(range(number_of_colors)))
    i=0
    for fn in tqdm(img_fns):
        fn_stem=Path(fn).stem
        fn_key,fn_idx=fn_stem.split("_")   
        
        img=get_image(fn)
        img_h,img_w,_=img.shape
        modified_img=cv2.resize(img, (int(img_w*resize_scale),int(img_h*resize_scale),), interpolation = cv2.INTER_AREA)
        modified_img=modified_img.reshape(modified_img.shape[0]*modified_img.shape[1], 3)
        clf=KMeans(n_clusters=number_of_colors)
        labels=clf.fit_predict(modified_img)
        
        counts=Counter(labels)
        center_colors=clf.cluster_centers_
        ordered_colors=[center_colors[i] for i in counts.keys()] # We get ordered colors by iterating through the keys
        hex_colors=[RGB2HEX(ordered_colors[i]) for i in counts.keys()]
        rgb_colors=[ordered_colors[i] for i in counts.keys()]
        
        if (show_chart):
            plt.figure(figsize = (8, 6))
            plt.pie(counts.values(), labels=hex_colors, colors=hex_colors)       
        
        coord=coords[fn_key][int(fn_idx)]
        color_dic={k:hex_colors[k] for k in range(number_of_colors) }    
        color_dic.update({"fn_stem":fn_stem,"fn_key":fn_key,"fn_idx":int(fn_idx),"geometry":Point(coord)})    
        img_dominant_color=img_dominant_color.append(color_dic,ignore_index=True)        
        
        # break
        if i==2:break
        i+=1
        
    wgs84=pyproj.CRS('EPSG:4326')
    img_dominant_color_gdf=gpd.GeoDataFrame(img_dominant_color,geometry=img_dominant_color.geometry,crs=wgs84) 
    
    return img_dominant_color_gdf

def find_dominant_colors_pool(fn,args):
    '''
    计算图像的主题色(多进程调用)

    Parameters
    ----------
    fn : string
        图像路径.
    args : list
        包括coords,resize_scale,number_of_colors.

    Returns
    -------
    color_dic : dict
        图像的主题色.

    '''    
    coords,resize_scale,number_of_colors=args
    fn_stem=Path(fn).stem
    fn_key,fn_idx=fn_stem.split("_")   
    
    img=get_image(fn)
    img_h,img_w,_=img.shape
    modified_img=cv2.resize(img, (int(img_w*resize_scale),int(img_h*resize_scale),), interpolation = cv2.INTER_AREA)
    modified_img=modified_img.reshape(modified_img.shape[0]*modified_img.shape[1], 3)
    clf=KMeans(n_clusters=number_of_colors)
    labels=clf.fit_predict(modified_img)
    
    counts=Counter(labels)
    center_colors=clf.cluster_centers_
    ordered_colors=[center_colors[i] for i in counts.keys()] # We get ordered colors by iterating through the keys
    hex_colors=[RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors=[ordered_colors[i] for i in counts.keys()]
    
    coord=coords[fn_key][int(fn_idx)]
    color_dic={k:hex_colors[k] for k in range(number_of_colors) }    
    color_dic.update({"fn_stem":fn_stem,"fn_key":fn_key,"fn_idx":int(fn_idx),"geometry":Point(coord)})    
    
    return color_dic

def dominant2cluster_colors(imgs_root,coords,resize_scale=0.5,number_of_colors=10,show_chart=False):
    '''
    主题色聚类

    Parameters
    ----------
    imgs_root : string
        图像所在根目录.
    coords : dict
        各个道路对应全景图的采集坐标点.
    resize_scale : numerical val, optional
        调整图像大小的比例. The default is 0.5.
    number_of_colors : int, optional
        主题色提取的数量. The default is 10.
    show_chart : bool, optional
        是否打印主题色的饼状图. The default is False.

    Returns
    -------
    None.

    '''    
    img_fns=glob.glob(os.path.join(imgs_root,'*.jpg'))
    img_dominant_color=pd.DataFrame(columns=["fn_stem","fn_key","fn_idx","geometry",]+list(range(number_of_colors)))
    i=0
    for fn in tqdm(img_fns):
        fn_stem=Path(fn).stem
        fn_key,fn_idx=fn_stem.split("_")   
        print('\n',fn_stem)
        
        img=get_image(fn)
        img_h,img_w,_=img.shape
        modified_img_w,modified_img_h=int(img_w*resize_scale),int(img_h*resize_scale),
        modified_img=cv2.resize(img, (modified_img_w,modified_img_h), interpolation = cv2.INTER_AREA)
        modified_img=modified_img.reshape(modified_img.shape[0]*modified_img.shape[1], 3)
        clf=KMeans(n_clusters=number_of_colors)
        labels=clf.fit_predict(modified_img)
        center_colors=clf.cluster_centers_
        labels_RGB=np.array([center_colors[i] for i in labels])
        labels_RGB_restore=labels_RGB.reshape((modified_img_h,modified_img_w,3))     
        plt.imshow(labels_RGB_restore/255)
        plt.show()
        
        labels_restored=labels.reshape((modified_img_h,modified_img_w,))
        plt.imshow(labels_restored,cmap="gist_ncar")      
        plt.show()
        
        counts=Counter(labels)
        ordered_colors=[center_colors[i] for i in counts.keys()] # We get ordered colors by iterating through the keys
        hex_colors=[RGB2HEX(ordered_colors[i]) for i in counts.keys()]
        if (show_chart):
            plt.figure(figsize = (8, 6))
            plt.pie(counts.values(), labels=hex_colors, colors=hex_colors)   
            plt.show()

        
        from skimage import measure
        img_labeled = measure.label(labels_restored, connectivity=1)
        plt.imshow(img_labeled,cmap="gist_ncar")
        plt.show()

        if i==0:break
        i+=1 

def dominant2cluster_colors_pool(fn,args):
    '''
    主题色聚类(多进程调用)

    Parameters
    ----------
    fn : string
        图像路径.
    args : list
        包括coords,resize_scale,number_of_colors.

    Returns
    -------
    color_dic : dict
        主题色聚类信息.

    '''    
    coords,resize_scale,number_of_colors=args
    fn_stem=Path(fn).stem
    fn_key,fn_idx=fn_stem.split("_")
    
    img=get_image(fn)
    img=img[:int(img.shape[0]*(70/100))]

    img_h,img_w,_=img.shape
    modified_img_w,modified_img_h=int(img_w*resize_scale),int(img_h*resize_scale),
    modified_img=cv2.resize(img, (modified_img_w,modified_img_h), interpolation = cv2.INTER_AREA)
    modified_img=modified_img.reshape(modified_img.shape[0]*modified_img.shape[1], 3)
    clf=KMeans(n_clusters=number_of_colors)
    labels=clf.fit_predict(modified_img)
    center_colors=clf.cluster_centers_
    
    labels_restored=labels.reshape((modified_img_h,modified_img_w,))    
    img_labeled=measure.label(labels_restored, connectivity=1)

    counts=Counter(img_labeled.flatten())
    coord=coords[fn_key][int(fn_idx)]
    color_dic={"fn_stem":fn_stem,"fn_key":fn_key,"fn_idx":int(fn_idx),"geometry":Point(coord),'counter':dict(counts)} 

    return color_dic






















    

