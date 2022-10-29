# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 15:55:36 2022

@author: richie bao
"""
import cv2 as cv
from tqdm import tqdm

class DynamicStreetView_visualPerception:
    '''
    class - 应用Star提取图像关键点，结合SIFT获得描述子，根据特征匹配分析特征变化（视觉感知变化），即动态街景视觉感知
    
    Params:
        imgs_fp - 图像路径列表；list(string)
        knnMatch_ratio - 图像匹配比例，默认为0.75；float
    '''
    
    def __init__(self,imgs_fp,knnMatch_ratio=0.75):
        self.knnMatch_ratio=knnMatch_ratio
        self.imgs_fp=imgs_fp
    
    def kp_descriptor(self,img_fp):        
        '''
        function - 提取关键点和获取描述子
        '''        
        
        img=cv.imread(img_fp)
        star_detector=cv.xfeatures2d.StarDetector_create()        
        key_points=star_detector.detect(img) # 应用处理Star特征检测相关函数，返回检测出的特征关键点
        img_gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY) # 将图像转为灰度
        kp,des=cv.xfeatures2d.SIFT_create().compute(img_gray, key_points) # SIFT特征提取器提取特征
        return kp,des        
     
    def feature_matching(self,des_1,des_2,kp_1=None,kp_2=None):        
        '''
        function - 图像匹配
        '''
        
        bf=cv.BFMatcher()
        matches=bf.knnMatch(des_1,des_2,k=2)
        
        '''
        可以由匹配matches返回关键点（train,query）的位置索引，train图像的索引，及描述子之间的距离
        DMatch.distance - Distance between descriptors. The lower, the better it is.
        DMatch.trainIdx - Index of the descriptor in train descriptors
        DMatch.queryIdx - Index of the descriptor in query descriptors
        DMatch.imgIdx - Index of the train image.
        '''
        '''
        if kp_1 !=None and kp_2 != None:
            kp1_list=[kp_1[mat[0].queryIdx].pt for mat in matches]
            kp2_list=[kp_2[mat[0].trainIdx].pt for mat in matches]
            des_distance=[(mat[0].distance,mat[1].distance) for mat in matches]
            print(des_distance[:5])
        '''
        
        good=[]
        for m,n in matches:
            if m.distance < self.knnMatch_ratio*n.distance:
                good.append(m) 

        return good 
    
    def sequence_statistics(self):        
        '''
        function - 序列图像匹配计算，每一位置图像与后续所有位置匹配分析
        '''                
        
        des_list=[]
        print("计算关键点和描述子...")
        for f in tqdm(self.imgs_fp):        
            _,des=self.kp_descriptor(f)
            des_list.append(des)
        matches_sequence={}
        print("计算序列图像匹配数...")
        for i in tqdm(range(len(des_list)-1)):
            matches_temp=[]
            for j_des in des_list[i:]:
                matches_temp.append(self.feature_matching(des_list[i],j_des))
            matches_sequence[i]=matches_temp
        matches_num={k:[len(v) for v in val] for k,val in matches_sequence.items()}
        return matches_num  