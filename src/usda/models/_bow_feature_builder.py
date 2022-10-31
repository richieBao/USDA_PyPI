# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 14:54:49 2022

@author: richie bao
"""
import cv2 as cv
from sklearn.cluster import KMeans
from tqdm import tqdm
import numpy as np

class feature_builder_BOW:
    '''
    class - 根据所有图像关键点描述子聚类建立图像视觉词袋，获取每一图像的特征（码本）映射的频数统计
    '''   
    def __init__(self,num_cluster=32):
        self.num_clusters=num_cluster

    def extract_features(self,img):        
        '''
        function - 提取图像特征
        
        Params:
            img - 读取的图像
        '''        
        
        star_detector=cv.xfeatures2d.StarDetector_create()
        key_points=star_detector.detect(img)
        img_gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        kp,des=cv.xfeatures2d.SIFT_create().compute(img_gray, key_points) # SIFT特征提取器提取特征
        return des
    
    def visual_BOW(self,des_all):        
        '''
        function - 聚类所有图像的特征（描述子/SIFT），建立视觉词袋
        
        Params:
            des_all - 所有图像的关键点描述子
        '''        
        
        print("start KMean...")
        kmeans=KMeans(self.num_clusters)
        kmeans=kmeans.fit(des_all)
        # centroids=kmeans.cluster_centers_
        print("end KMean...")
        return kmeans         
    
    def get_visual_BOW(self,training_data):
        '''
        function - 提取图像特征，返回所有图像关键点聚类视觉词袋
        
        Params:
            training_data - 训练数据集
        '''        
        
        des_all=[]      
        for item in tqdm(training_data):            
            classi_judge=item['object_class']
            img=cv.imread(item['image_path'])
            des=self.extract_features(img)
            des_all.extend(des)     
        kmeans=self.visual_BOW(des_all)      
        return kmeans
    
    def normalize(self,input_data):        
        '''
        fuction - 归一化数据
        
        Params:
            input_data - 待归一化的数组
        '''        
        
        sum_input=np.sum(input_data)
        if sum_input>0:
            return input_data/sum_input # 单一数值/总体数值之和，最终数值范围[0,1]
        else:
            return input_data               
    
    def construct_feature(self,img,kmeans):        
        '''
        function - 使用聚类的视觉词袋构建图像特征（构造码本）
        
        Paras:
            img - 读取的单张图像
            kmeans - 已训练的聚类模型
        '''
        
        des=self.extract_features(img)
        labels=kmeans.predict(des.astype(float)) # 对特征执行聚类预测类标
        feature_vector=np.zeros(self.num_clusters)
        for i,item in enumerate(feature_vector): # 计算特征聚类出现的频数/直方图
            feature_vector[labels[i]]+=1
        feature_vector_=np.reshape(feature_vector,((1,feature_vector.shape[0])))
        return self.normalize(feature_vector_)
    
    def get_feature_map(self,training_data,kmeans):        
        '''
        function - 返回每个图像的特征映射（码本映射）
        
        Paras:
            training_data - 训练数据集
            kmeans - 已训练的聚类模型
        '''
        
        feature_map=[]
        for item in training_data:
            temp_dict={}
            temp_dict['object_class']=item['object_class']
            img=cv.imread(item['image_path'])
            temp_dict['feature_vector']=self.construct_feature(img,kmeans)
            if temp_dict['feature_vector'] is not None:
                feature_map.append(temp_dict)

        return feature_map