# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 16:36:32 2022

@author: richie bao
"""
from sklearn import preprocessing
import pickle  
import numpy as np
from ..models import feature_builder_BOW

class ImageTag_extractor:
    '''
    class - 图像识别器，基于图像分类模型，视觉词袋以及图像特征
    '''
    def __init__(self, ERF_clf_fp, visual_BOW_fp,visual_feature_fp):      
        with open(ERF_clf_fp,'rb') as f:  # 读取存储的图像分类器模型
            self.clf=pickle.load(f)

        with open(visual_BOW_fp,'rb') as f:  # 读取存储的聚类模型和聚类中心点
            self.kmeans=pickle.load(f)

        '''对标签编码'''
        with open(visual_feature_fp, 'rb') as f:
            self.feature_map=pickle.load(f)
        self.label_words=[x['object_class'] for x in self.feature_map]
        self.le=preprocessing.LabelEncoder()
        self.le.fit(self.label_words)   
        
    def predict(self,img):
        feature_vector=feature_builder_BOW().construct_feature(img,self.kmeans)  # 提取图像特征，之前定义的feature_builder_BOW()类，可放置于util.py文件中，方便调用
        label_nums=self.clf.predict(np.asarray(feature_vector)) # 进行图像识别/分类
        image_tag=self.le.inverse_transform([int(x) for x in label_nums])[0] # 获取图像分类标签
        return image_tag
    
    