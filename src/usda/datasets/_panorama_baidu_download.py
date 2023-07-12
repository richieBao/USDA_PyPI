# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 14:46:58 2021
updated on Wed Jan 19 16:00:24 2022

@author: Richie Bao-caDesign设计(cadesign.cn)
"""
from tqdm import tqdm
import numpy as np
from shapely.geometry import MultiPoint
import pyproj
from shapely.ops import transform
import geopandas as gpd

import urllib,os
import pickle

from PIL import Image

import glob
from PIL import Image
import shutil

def roads_pts4bsv(roads_gdf,distance=10):
    '''
    给定GeoDataFrame的道路中心线，和采样距离，返回采样点

    Parameters
    ----------
    roads_gdf : GeoDataFrame
        GeoDataFrame的道路中心线.
    distance : numerical value, optional
        采样距离. The default is 10.

    Returns
    -------
    GeoDataFrame
        采样点.

    '''
    
    tqdm.pandas()    
    def line_pts(line):
        dists=np.arange(0,line.length,distance)
        pts=MultiPoint([line.interpolate(d,normalized=False) for d in dists])
        return pts      
        
    roads_gdf['pts']=roads_gdf.geometry.progress_apply(line_pts)
    
    wgs84=pyproj.CRS('EPSG:4326')
    utm=roads_gdf.crs #pyproj.CRS(roads_gpd.crs.srs)
    project=pyproj.Transformer.from_crs(utm, wgs84, always_xy=True).transform
    roads_gdf['pts_wgs84']=roads_gdf.pts.progress_apply(lambda row:transform(project,row))    
    
    pts_gdf=gpd.GeoDataFrame(roads_gdf[['Name','Uid']],geometry=roads_gdf.pts_wgs84.to_list(),crs=wgs84) #roads_gdf.drop(['geometry'],axis=1)    
    return pts_gdf

   
def baidu_steetview_crawler(pts_gdf,save_path,ak,save_path_BSV_retrival_info):
    '''
    从百度地图应用中，根据采样点检索下载全景图
    
    Parameters
    ----------
    pts_gdf : GeoDataFrame
        采样点.
    save_path : string
        全景图的保存路径.
    ak : string
        访问应用的AK值，在百度应用中注册申请.
    save_path_BSV_retrival_info : dict
        pickle方式保存下载信息，包括pt_fns-以路径为键，全景图下载地址列表为值；coords-以路径为键，采样点坐标列表为值；
                                 downloadError_idx-错误索引列表.

    Returns
    -------
    coords : dict
        以路径为键，值为采样点坐标列表.
    pts_num : int
        下载全景图的数量.
        
    '''    
    downloadError_idx=[]
    coords={}
    pts_num={}    
    pt_fns={}
    for idx,row in pts_gdf.iterrows():
        pt_coords=[(pt.x,pt.y) for pt in row.geometry]
        coords[row.Name]=pt_coords
        pts_num[row.Name]=len(pt_coords)
    print("\npts_num={}".format(sum(pts_num.values())))
    
    urlRoot=r"http://api.map.baidu.com/panorama/v2?"
    query_dic={
        'width':'1024',
        'height':'512', 
        'fov':'360',
        'heading':'0',
        'pitch':'0',
        'coordtype':'wgs84ll',
        'ak':ak,
    }   
    # tt=0
    for k,v in tqdm(coords.items()):
        pt_fn=[]
        for i,coord in enumerate(v):
            pic_fn=os.path.join(save_path,"{}_{}.jpg".format(k,i))                        
            if not os.path.exists(pic_fn):
                #update query arguments
                query_dic.update({
                                  'location':str(coord[0])+','+str(coord[1]),
                                 })         
                url=urlRoot+urllib.parse.urlencode(query_dic)
                try:
                    data=urllib.request.urlopen(url)
                    pt_fn.append(pic_fn)
                    with open(pic_fn,'wb') as fp:
                        fp.write(data.read())           
                except:
                    downloadError_idx.append((k,i))
                    print('download_error:{},{}'.format(k,i))
            else:
                print("file existed.")
                
        pt_fns[k]=pt_fn
        # if tt==2:break
        # tt+=1
        
    with open(save_path_BSV_retrival_info["pt_fns"],'wb') as f:
        pickle.dump(pt_fns,f)
    with open(save_path_BSV_retrival_info["coords"],'wb') as f:
        pickle.dump(coords,f)       
    with open(save_path_BSV_retrival_info["downloadError_idx"],'wb') as f:
        pickle.dump(downloadError_idx,f)      
            
    return coords,pts_num

def img_valid(img_fns,save_path_img_valid):
    '''
    验证图像是否有效，即是否可以被打开

    Parameters
    ----------
    img_fns : dict
        图像路径字典.
    save_path_img_valid : dict
        保存路径字典.

    Returns
    -------
    img_val : dict
        返回有效图像字典.

    '''    
    img_val={}
    img_inval=[]
    for k,v in tqdm(img_fns.items()):
        fns=[]
        for fn in v:
            try:
                im=Image.open(fn)
                fns.append(fn)
            except:
                img_inval.append(fn)        
        img_val[k]=fns
            
    with open(save_path_img_valid["img_val"],'wb') as f:
        pickle.dump(img_val,f)        
    with open(save_path_img_valid["img_inval"],'wb') as f:
        pickle.dump(img_inval,f)                 
    return img_val

def img_valid_copy_folder(imgs_root,panoramic_imgs_valid_root):
    '''
    功能基本同img_valid(img_fns,save_path_img_valid)函数。除了验证图像是否有效，同时将其复制到新建的文件夹下

    Parameters
    ----------
    imgs_root : string
        图像根目录.
    panoramic_imgs_valid_root : string
        新建文件夹，复制有效图像至该文件夹.

    Returns
    -------
    None.

    '''    
    img_fns=glob.glob(os.path.join(imgs_root,'*.jpg'))
    # print(img_fns)  
    img_val=[]
    img_inval=[]
    for fn in tqdm(img_fns):
        try:
            im=Image.open(fn)
            img_val.append(fn)
        except:
            img_inval.append(fn)    
    # print(img_val)
    for fn in tqdm(img_val):
        shutil.copy(fn,panoramic_imgs_valid_root)   
        
def roads_pts4bsv_tourLine(roads_gdf,distance=10):
    '''
    由道路GeoDataFrame数据，提取给定距离的采样点

    Parameters
    ----------
    roads_gdf : GeoDataFrame
        道路线.
    distance : numericle value, optional
        指定采样距离. The default is 10.

    Returns
    -------
    GeoDataFrame
        采样点.

    '''    
    tqdm.pandas()    
    def line_pts(line):
        dists=np.arange(0,line.length,distance)
        pts=MultiPoint([line.interpolate(d,normalized=False) for d in dists])
        return pts      
        
    roads_gdf['pts']=roads_gdf.geometry.progress_apply(line_pts)
    
    wgs84=pyproj.CRS('EPSG:4326')
    utm=roads_gdf.crs #pyproj.CRS(roads_gpd.crs.srs)
    project=pyproj.Transformer.from_crs(utm, wgs84, always_xy=True).transform
    roads_gdf['pts_wgs84']=roads_gdf.pts.progress_apply(lambda row:transform(project,row))    
    
    pts_gdf=gpd.GeoDataFrame(roads_gdf[['Name','group']],geometry=roads_gdf.pts_wgs84.to_list(),crs=wgs84) #roads_gdf.drop(['geometry'],axis=1)    
    return pts_gdf

def pts_number_check(pts):
    '''
    查看采样点，打印数量

    Parameters
    ----------
    pts : GeoDataFrame
        输入点数据.

    Returns
    -------
    None.

    '''
    
    downloadError_idx=[]
    coords={}
    pts_num={}    
    pt_fns={}
    for idx,row in pts.iterrows():
        pt_coords=[(pt.x,pt.y) for pt in row.geometry]
        coords[row.Name]=pt_coords
        pts_num[row.Name]=len(pt_coords)
    # print(coords)
    print("\npts_num={}".format(sum(pts_num.values())))
    
if __name__=="__main__":
    pass