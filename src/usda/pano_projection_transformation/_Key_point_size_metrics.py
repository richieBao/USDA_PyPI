# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 11:01:29 2023

@author: richie bao
"""
import cv2 as cv
from sklearn.cluster import KMeans
from tqdm import tqdm
import numpy as np
from pathlib import Path

import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
import pyproj

class feature_builder_BOW:
    '''
    class - 根据所有图像关键点描述子聚类建立图像视觉词袋，获取每一图像的特征（码本）映射的频数统计
    '''   
    def __init__(self,num_cluster=32):
        self.num_clusters=num_cluster

    def extract_features(self,img):        
        '''
        function - 提取图像特征
        
        Paras:
        img - 读取的图像
        '''
        star=cv.xfeatures2d.StarDetector_create() 
        key_point=star.detect(img)
        cv.drawKeypoints(img,key_point,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)        
        # Initiate BRIEF extractor
        brief=cv.xfeatures2d.BriefDescriptorExtractor_create()    
        # compute the descriptors with BRIEF
        kp, des=brief.compute(img, key_point)   

        return des,kp
    
    def visual_BOW(self,des_all):        
        '''
        function - 聚类所有图像的特征（描述子/SIFT），建立视觉词袋
        
        des_all - 所有图像的关键点描述子
        '''
        print("start KMean...")
        kmeans=KMeans(self.num_clusters)
        kmeans=kmeans.fit(des_all)
        print("end KMean...")
        return kmeans         
    
    def get_visual_BOW(self,training_data):
        '''
        function - 提取图像特征，返回所有图像关键点聚类视觉词袋
        
        Paras:
        training_data - 训练数据集
        '''
        des_all=[]
        # i=0        
        for item in tqdm(training_data):
            img=cv.imread(item)
            img=img[:int(img.shape[0]*(70/100))]  
            
            des,_=self.extract_features(img)
            des_all.extend(des)           

            # if i==10:break
            # i+=1        
        kmeans=self.visual_BOW(des_all)      
        return kmeans
    
    def normalize(self,input_data):        
        '''
        fuction - 归一化数据
        
        input_data - 待归一化的数组
        '''
        sum_input=np.sum(input_data)
        if sum_input>0:
            return input_data/sum_input #单一数值/总体数值之和，最终数值范围[0,1]
        else:
            return input_data               
    
    def construct_feature(self,img,kmeans):
        '''
        function - 使用聚类的视觉词袋构建图像特征（构造码本）
        
        Paras:
        img - 读取的单张图像
        kmeans - 已训练的聚类模型
        '''
        des,kp=self.extract_features(img)
        labels=kmeans.predict(des.astype(float)) #对特征执行聚类预测类标
        feature_vector=np.zeros(self.num_clusters)
        for i,item in enumerate(labels): #计算特征聚类出现的频数/直方图
            feature_vector[labels[i]]+=1
        feature_vector_=np.reshape(feature_vector,((1,feature_vector.shape[0])))
        return feature_vector_,labels,kp
    
    def get_feature_map(self,training_data,kmeans):
        '''
        function - 返回每个图像的特征映射（码本映射）
        Paras:
        training_data - 训练数据集
        kmeans - 已训练的聚类模型
        '''
        feature_map=[]
        for item in tqdm(training_data):            
            fn_stem=Path(item).stem
            fn_key,fn_idx=fn_stem.split("_")
            
            temp_dict={}
            temp_dict['fn_stem']=fn_stem
            img=cv.imread(item)
            img=img[:int(img.shape[0]*(70/100))]  
            
            feature_vector,labels,kp=self.construct_feature(img,kmeans)
            temp_dict['feature_vector']=feature_vector
            temp_dict['labels']=labels
            temp_dict['kp']=[{'pt':kp[i].pt, 
                             'size':kp[i].size, 
                             'angle':kp[i].angle,
                             'response':kp[i].response, 
                             'octave':kp[i].octave, 
                             'class_id':kp[i].class_id} for i in range(len(kp))]
            if temp_dict['feature_vector'] is not None:
                feature_map.append(temp_dict)
        return feature_map

def kps_desciptors_BOW_feature(feature_map,coords,num_cluster=32):
    '''
    码本映射转换为GeoDataFrame

    Parameters
    ----------
    feature_map : list
        图像的特征映射（码本映射）.
    coords : dict
        各个道路对应全景图的采集坐标点.
    num_cluster : int, optional
        聚类的数量. The default is 32.

    Returns
    -------
    featureMap_gdf : GeoDataFrame
        码本映射(GDF格式).

    '''    
    # panorama_object_df=pd.DataFrame(columns=["fn_stem","fn_key","fn_idx","geometry",]+list(range(num_cluster)))
    panorama_object_df_lst=[]
    for feature_info in tqdm(feature_map):
        fn_stem=feature_info['fn_stem']
        fn_key,fn_idx=fn_stem.split("_")
        featureMap_dict=dict(zip(list(range(num_cluster)),feature_info['feature_vector'].tolist()[0]))
        coord=coords[fn_key][int(fn_idx)]
        featureMap_dict.update({"fn_stem":fn_stem,"fn_key":fn_key,"fn_idx":int(fn_idx),"geometry":Point(coord)})   
        # panorama_object_df=panorama_object_df.append(featureMap_dict,ignore_index=True)
        panorama_object_df_lst.append(featureMap_dict)
        # break
    
    panorama_object_df=pd.DataFrame(panorama_object_df_lst,columns=["fn_stem","fn_key","fn_idx","geometry",]+list(range(num_cluster)))
    wgs84=pyproj.CRS('EPSG:4326')
    featureMap_gdf=gpd.GeoDataFrame(panorama_object_df,geometry=panorama_object_df.geometry,crs=wgs84)     
    
    return featureMap_gdf

def kp_stats(feature_map,coords,bins): 
    '''
    给定分割区间，统计关键点邻域尺度

    Parameters
    ----------
    feature_map : list
        图像的特征映射（码本映射）.
    coords : dict
        各个道路对应全景图的采集坐标点.
    bins : list
        分割区间.

    Returns
    -------
    kp_size_stats_gdf : GeoDataFrame
        统计关键点邻域统计.

    '''    
    kp_dict={i['fn_stem']:i['kp'] for i in feature_map}
    i=0
    size_stats_dict_list=[]
    bins=bins
    for fn_stem,v in tqdm(kp_dict.items()):
        kp_df=pd.DataFrame(v)
        size_stats_dict=kp_df['size'].describe().to_dict()
        size_stats_dict['num']=len(v)        
        
        size_stats_dict['fn_stem']=fn_stem
        fn_key,fn_idx=fn_stem.split("_")
        size_stats_dict['fn_key']=fn_key
        size_stats_dict['fn_idx']=fn_idx
        
        coord=coords[fn_key][int(fn_idx)]
        size_stats_dict['geometry']=Point(coord)
        # print(size_stats_dict)        
        
        # print(kp_df['size'])
        fre_size=kp_df[['size']].apply(pd.Series.value_counts,bins=bins,).to_dict()['size']
        fre_size={'{}_{}'.format(k.left,k.right):v for k,v in fre_size.items()}
        # print(fre_size)
        size_stats_dict.update(fre_size)
        
        size_stats_dict_list.append(size_stats_dict)
        # if i==10:break
        # i+=1
    kp_size_stats_df=pd.DataFrame.from_dict(size_stats_dict_list)
    wgs84=pyproj.CRS('EPSG:4326')
    kp_size_stats_gdf=gpd.GeoDataFrame(kp_size_stats_df,geometry=kp_size_stats_df.geometry,crs=wgs84) 
    return kp_size_stats_gdf 

if __name__=="__main__":
    import glob,os
    num_cluster=32
    img_fp_list=glob.glob(os.path.join('G:\\data\\pano_dongxistreet\\images_valid','*.jpg'))
    kmeans=feature_builder_BOW(num_cluster).get_visual_BOW(img_fp_list)    
        
        
        
    
    
#     import pickle
#     import glob
#     import sys,os
#     sys.path.append('..')      
#     from database import postSQL2gpd,gpd2postSQL,cfg_load_yaml 
#     parent_path=os.path.dirname(os.getcwd())
#     cfg=cfg_load_yaml('../config.yml')  
    
#     UN=cfg["postgreSQL"]["myusername"]
#     PW=cfg["postgreSQL"]["mypassword"]
#     DB=cfg["postgreSQL"]["mydatabase"]
    
#     img_path=cfg['streetview']['panoramic_imgs_valid_root']    
#     img_fp_list=glob.glob(os.path.join(img_path,'*.jpg'))
    
#     #A.提取图像特征，返回所有图像关键点聚类视觉词袋    
#     num_cluster=cfg['KP_metrics']['num_cluster']
#     kmeans=feature_builder_BOW(num_cluster).get_visual_BOW(img_fp_list)  
#     visual_BOW_region_fn=os.path.join(parent_path,cfg['KP_metrics']['visual_BOW_region_fn'])

#     with open(visual_BOW_region_fn,'wb') as f: 
#         pickle.dump(kmeans,f) #存储kmeans聚类模型    
    
#     #B.图像的特征映射（码本映射）
#     with open(visual_BOW_region_fn,'rb') as f:
#         kmeans=pickle.load(f)    
#     feature_map=feature_builder_BOW(num_cluster).get_feature_map(img_fp_list,kmeans) 
#     feature_map_region_fn=cfg['KP_metrics']['feature_map_region_fn']    
#     with open(feature_map_region_fn,'wb') as f: 
#         pickle.dump(feature_map,f)  
    
#     #C.码本映射转换为GeoDataFrame，写入数据库
#     with open(feature_map_region_fn,'rb') as f:
#         feature_map_region=pickle.load(f)     
#     with open(os.path.join(parent_path,cfg['streetview']['save_path_BSV_retrival_info']['coords']),'rb') as f: 
#         coords=pickle.load(f)       
#     featureMap_gdf=kps_desciptors_BOW_feature(feature_map_region,coords,num_cluster) 
#     featureMap_table_name=cfg['KP_metrics']['featureMap_table_name']
#     gpd2postSQL(featureMap_gdf,table_name=featureMap_table_name,myusername=UN,mypassword=PW,mydatabase=DB)
    
#     #D.给定分割区间，统计关键点邻域尺度 
#     bins=cfg['KP_metrics']['bins']
#     kp_size_stats_gdf=kp_stats(feature_map_region,coords,bins)
#     kp_size_stats_table_name=cfg['KP_metrics']['kp_size_stats_table_name']
#     gpd2postSQL(kp_size_stats_gdf,table_name=kp_size_stats_table_name,myusername=UN,mypassword=PW,mydatabase=DB)