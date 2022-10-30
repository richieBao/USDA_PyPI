# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 20:10:32 2022

@author: richie bao
"""
import os,time,warnings
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn import cluster, datasets, mixture
from itertools import cycle, islice   

from sklearn import datasets
from numpy.random import rand

def img_theme_color(imgs_root,imgsFn_lst,columns,scale,):       
    '''
    function - 聚类的方法提取图像主题色，并打印图像、聚类预测类的二维显示和主题色带
    
    Params:
        imgs_root - 图像所在根目录；string
        imgsFn_lst - 图像名列表；list(string)
        columns - 列数；int    
        
    Returns:
        themes - 图像主题色；array
        pred - 预测的类标；array
    ''' 
    
    # 设置聚类参数，本实验中仅使用了KMeans算法，其它算法可以自行尝试
    kmeans_paras={'quantile': .3,
                  'eps': .3,
                  'damping': .9,
                  'preference': -200,
                  'n_neighbors': 10,
                  'n_clusters': 7}     
        
    imgsPath_lst=[os.path.join(imgs_root,p) for p in imgsFn_lst]
    imgs_rescale=[(img_rescale(img,scale)) for img in imgsPath_lst]  
    datasets=[((i[1],None),{}) for i in imgs_rescale] # 基于img_2d的图像数据，用于聚类计算
    img_lst=[i[0] for i in imgs_rescale]  # 基于img_3d的图像数据，用于图像显示
    
    themes=np.zeros((kmeans_paras['n_clusters'], 3))  # 建立0占位的数组，用于后面主题数据的追加。'n_clusters'为提取主题色的聚类数量，此处为7，轴2为3，是色彩的RGB数值
    (img_3d,img_2d)=imgs_rescale[0]  # 可以1次性提取元组索引值相同的值，img就是img_3d，而pix是img_2d
    img2d_V,img2d_H=img_2d.shape  # 获取img_2d数据的形状，用于pred预测初始数组的建立
    pred=np.zeros((img2d_V))  # 建立0占位的pred预测数组，用于后面预测结果数据的追加，即图像中每一个像素点属于设置的7个聚类中的哪一组，预测给定类标
    
    plt.figure(figsize=(6*3+3, len(imgsPath_lst)*2))  # 图表大小的设置，根据图像的数量来设置高度，宽度为3组9个子图，每组包括图像、预测值散点图和主题色
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.3,hspace=.3)  # 调整图，避免横纵向坐标重叠    
    subplot_num=1  # 子图的计数  
    
    for i_dataset, (dataset, algo_params) in tqdm(enumerate(datasets)):  # 循环pixData数据，即待预测的每个图像数据。enumerate()函数将可迭代对象组成一个索引序列，可以同时获取索引和值，其中i_dataset为索引，从整数0开始
        X, y=dataset  # 用于机器学习的数据一般包括特征值和类标，此次实验为无监督分类的聚类实验，没有类标，并将其在前文中设置为None对象
        Xstd=StandardScaler().fit_transform(X)  # 标准化数据仅用于二维图表的散点，可视化预测值，而不用于聚类，聚类数据保持色彩的0-255值范围
        # 此次实验使用KMeans算法，参数为'n_clusters'一项。不同算法计算效率不同，例如MiniBatchKMeans和KMeans算法计算较快
        km=cluster.KMeans(n_clusters=kmeans_paras['n_clusters'])
        clustering_algorithms=(('KMeans',km),)
        for name, algorithm in clustering_algorithms: 
            t0=time.time()  
            # 警告错误，使用warning库
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="the number of connected components of the " +"connectivity matrix is [0-9]{1,2}" +" > 1. Completing it to avoid stopping the tree early.",
                    category=UserWarning)
                warnings.filterwarnings(
                    "ignore",
                    message="Graph is not fully connected, spectral embedding" +" may not work as expected.",
                    category=UserWarning)
                algorithm.fit(X)  # 通过fit函数执行聚类算法            
        
            quantize=np.array(algorithm.cluster_centers_, dtype=np.uint8) # 返回聚类的中心，为主题色
            themes=np.vstack((themes,quantize))  # 将计算获取的每一图像主题色追加到themes数组中
            t1=time.time()  # 计算聚类算法所需时间
            '''获取预测值/分类类标'''   
            if hasattr(algorithm, 'labels_'):
                y_pred=algorithm.labels_.astype(int)
            else:
                y_pred=algorithm.predict(X)  
            pred=np.hstack((pred,y_pred))  # 将计算获取的每一图像聚类预测结果追加到pred数组中
            fig_width=(len(clustering_algorithms)+2)*3  # 水平向子图数
            plt.subplot(len(datasets), fig_width,subplot_num)
            plt.imshow(img_lst[i_dataset])  # 图像显示子图
            
            plt.subplot(len(datasets),fig_width, subplot_num+1)
            if i_dataset == 0:
                plt.title(name, size=18)
            colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a','#f781bf', '#a65628', '#984ea3','#999999', '#e41a1c', '#dede00']),int(max(y_pred) + 1))))  # 设置预测类标分类颜色
            plt.scatter(Xstd[:, 0], Xstd[:, 1], s=10, color=colors[y_pred]) # 预测类标子图
            plt.xlim(-2.5, 2.5)
            plt.ylim(-2.5, 2.5)
            plt.xticks(())
            plt.yticks(())
            plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),transform=plt.gca().transAxes, size=15,horizontalalignment='right')  # 子图中显示聚类计算时间长度，     
            # 图像主题色子图参数配置
            plt.subplot(len(datasets), fig_width,subplot_num+2)
            t=1
            pale=np.zeros(img_lst[i_dataset].shape, dtype=np.uint8)
            h, w,_=pale.shape
            ph=h/len(quantize)
            for y in range(h):
                pale[y,::] = np.array(quantize[int(y/ph)], dtype=np.uint8)
            plt.imshow(pale)    
            t+=1  
            subplot_num+=3    
    plt.show()            
    return themes,pred

def themeColor_impression(theme_color):
    '''
    function - 显示所有图像主题色，获取总体印象
    
    Params:
        theme_color - 主题色数组；array
        
    Returns:
        None
    '''  
    
    n_samples=theme_color.shape[0]
    random_state=170  #可 为默认，不设置该参数，获得随机图形
    # 利用scikit的datasets数据集构建有差异变化的斑点
    varied=datasets.make_blobs(n_samples=n_samples,cluster_std=[1.0, 2.5, 0.5],random_state=random_state)
    (x,y)=varied    
    fig, ax=plt.subplots(figsize=(10,10))
    scale=1000.0*rand(n_samples)  # 设置斑点随机大小
    ax.scatter(x[...,0], x[...,1], c=theme_color/255,s=scale,alpha=0.7, edgecolors='none')  # 将主题色赋予斑点
    ax.grid(True)       
    plt.show()  