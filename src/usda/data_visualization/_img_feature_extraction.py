# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 14:00:49 2022

@author: richie bao
"""
import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt 
import copy

def Gaussion_blur(img_fp):       
    '''
    function - 应用OpenCV计算高斯模糊，并给定滑动条调节参数
    
    Params:
        img_fp - 图像路径；string
    
    Returns:
        None
    '''    
    
    # 回调函数
    def Gaussian_blur_size(GBlur_size): # 高斯核(卷积核大小)，值越大，图像越模糊
        global KSIZE 
        KSIZE = GBlur_size * 2 +3
        print("changes in kernel size:",KSIZE, SIGMA)
        dst = cv.GaussianBlur(img, (KSIZE,KSIZE), SIGMA, KSIZE) 
        cv.imshow(window_name,dst)

    def Gaussian_blur_Sigma(GBlur_sigma): # σ(sigma)设置，值越大，图像越模糊
        global SIGMA
        SIGMA = GBlur_sigma/10.0
        print ("changes in sigma:",KSIZE, SIGMA)
        dst = cv.GaussianBlur(img, (KSIZE,KSIZE), SIGMA, KSIZE) 
        cv.imshow(window_name,dst)

    # 全局变量
    GBlur_size = 1
    GBlur_sigma = 15
    
    global KSIZE 
    global SIGMA
    KSIZE = 1
    SIGMA = 15
    max_value = 300
    max_type = 6
    window_name = "Gaussian Blur"
    trackbar_size = "Size*2+3"
    trackbar_sigema = "Sigma/10"

    # 读入图片，模式为灰度图，创建窗口
    img= cv.imread(img_fp,0)
    cv.namedWindow(window_name)
   
    # 创建滑动条
    cv.createTrackbar( trackbar_size, window_name,GBlur_size, max_type,Gaussian_blur_size)
    cv.createTrackbar( trackbar_sigema, window_name,GBlur_sigma, max_value, Gaussian_blur_Sigma)    
    
    # 初始化
    Gaussian_blur_size(GBlur_size)
    Gaussian_blur_Sigma(GBlur_sigma)        

    if cv.waitKey(0) == 27:  
        cv.destroyAllWindows()   
        
def SIFT_detection(img_fp,save=False):    
    '''
    function - 尺度不变特征变换(scale invariant feature transform，SIFT)特征点检测
    
    Params:
        img_fp - 图像文件路径；string
        save- 是否保存特征结果图像。 The default is False；bool
        
    Returns:
        None
    '''
       
    img=cv.imread(img_fp)
    img_gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    sift=cv.SIFT_create() # SIFT特征实例化 cv.xfeatures2d.SIFT_create()
    key_points=sift.detect(img_gray,None)  # 提取SIFT特征关键点detector
    
    # 示例打印关键点数据
    for k in key_points[:5]:
        print("关键点点坐标:%s,直径:%.3f,金字塔层:%d,响应程度:%.3f,分类:%d,方向:%d"%(k.pt,k.size,k.octave,k.response,k.class_id,k.angle))
        """
        关键点信息包含：
        k.pt关键点点的坐标(图像像素位置)
        k.size该点范围的大小（直径）
        k.octave从高斯金字塔的哪一层提取得到的数据
        k.response响应程度，代表该点强壮大小，即角点的程度。角点：极值点，某方面属性特别突出的点(最大或最小)。
        k.class_id对图像进行分类时，可以用class_id对每个特征点进行区分，未设置时为-1
        k.angle角度，关键点的方向。SIFT算法通过对邻域做梯度运算，求得该方向。-1为初始值        
        """
    print("_"*50) 
    descriptor=sift.compute(img_gray,key_points) # 提取SIFT调整描述子-descriptor，返回的列表长度为2，第1组数据为关键点，第2组数据为描述子(关键点周围对其有贡献的像素点)
    print("key_points数据类型:%s,descriptor数据类型:%s"%(type(key_points),type(descriptor)))
    print("关键点：")
    print(descriptor[0][:1]) # 关键点
    print("描述子：")
    print(descriptor[1][:1]) # 描述子
    print("描述子 shape:",descriptor[1].shape)      
    
    cv.drawKeypoints(img,key_points,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
    
    if save:
        cv.imshow('sift features',img)
        cv.imwrite('./data/sift_features.jpg',img) # 保存图像
        cv.waitKey()
    else:        
        fig, ax=plt.subplots(figsize=(30,15))
        ax.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB) )
        plt.show()         
        
def STAR_detection(img_fp,save=False):
    '''
    function - 使用Star特征检测器提取图像特征
    Params:
        img_fp - 图像文件路径  
        save- 是否保存特征结果图像。 The default is False；bool
        
    Returns:
        None
    '''    
    
    img=cv.imread(img_fp)
    star=cv.xfeatures2d.StarDetector_create() 
    key_point=star.detect(img)
    cv.drawKeypoints(img,key_point,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    if save:
        cv.imshow('star features',img_copy)
        cv.imwrite('./data/star_features.jpg',img) #保存图像
        cv.waitKey()
    else:        
        fig, ax=plt.subplots(figsize=(30,15))
        ax.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB) )
        plt.show()        
            
def feature_matching(img_1_fp,img_2_fp,index_params=None,method='FLANN'):   
    '''
    function - OpenCV 根据图像特征匹配图像。迁移官网的三种方法，1-ORB描述子蛮力匹配　Brute-Force Matching with ORB Descriptors；2－使用SIFT描述子和比率检测蛮力匹配 Brute-Force Matching with SIFT Descriptors and Ratio Test; 3-基于FLANN的匹配器 FLANN based Matcher
   
    Params:
        img_1 - 待匹配图像1路径；string
        img_2 - 待匹配图像2路径；string
        method - 参数为:'ORB','SIFT','FLANN'。The default is 'FLANN'；string
        
    Returns:
        None
    '''    
    
    plt.figure(figsize=(30,15))
    img1 = cv.imread(img_1_fp,cv.IMREAD_GRAYSCALE) # queryImage
    img2 = cv.imread(img_2_fp,cv.IMREAD_GRAYSCALE) # trainImage
    if method=='ORB':
        # Initiate ORB detector
        orb = cv.ORB_create()
        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(img1,None)
        kp2, des2 = orb.detectAndCompute(img2,None)
        
        # create BFMatcher object
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1,des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        # Draw first 10 matches.
        img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img3),plt.show()
        
    if method=='SIFT':        
        # Initiate SIFT detector
        sift = cv.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
        # BFMatcher with default params
        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1,des2,k=2)
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        # cv.drawMatchesKnn expects list of lists as matches.
        img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good[0:int(1*len(good)):int(0.1*len(good))],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img3),plt.show()        
    
    if method=='FLANN':
        # Initiate SIFT detector
        sift = cv.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]
        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                matchesMask[i]=[1,0]
        draw_params = dict(matchColor = (0,255,0),
                           singlePointColor = (255,0,0),
                           matchesMask = matchesMask,
                           flags = cv.DrawMatchesFlags_DEFAULT)
        img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
        plt.imshow(img3,),plt.show()        
        