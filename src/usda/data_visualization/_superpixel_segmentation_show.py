# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 15:17:39 2022

@author: richie bao
"""
import math
import os
import matplotlib.pyplot as plt
from PIL import Image
from skimage.segmentation import mark_boundaries  

def markBoundaries_layoutShow(segs_array,img,columns,titles,prefix,figsize=(15,10)):
    '''
    function - 给定包含多个图像分割的一个数组，排布显示分割图像边界。

    Paras:
        segs_array - 多个图像分割数组；ndarray
        img - 底图 ；ndarray
        columns - 列数；int
        titles - 子图标题；string
        figsize - 图表大小。The default is (15,10)；tuple
        
    Returns:
        None
    '''         
    
    rows=math.ceil(segs_array.shape[0]/columns)
    fig,axes=plt.subplots(rows,columns,sharex=True,sharey=True,figsize=figsize)   # 布局多个子图，每个子图显示一幅图像
    ax=axes.flatten()  # 降至1维，便于循环操作子图
    for i in range(segs_array.shape[0]):
        ax[i].imshow(mark_boundaries(img, segs_array[i]))  # 显示图像
        ax[i].set_title("{}={}".format(prefix,titles[i]))
    invisible_num=rows*columns-len(segs_array)
    if invisible_num>0:
        for i in range(invisible_num):
            ax.flat[-(i+1)].set_visible(False)
    fig.tight_layout() # 自动调整子图参数，使之填充整个图像区域  
    fig.suptitle("segs show",fontsize=14,fontweight='bold',y=1.02)
    plt.show()
    
def segMasks_layoutShow(segs_array,columns,titles,prefix,cmap='prism',figsize=(20,10)):
    '''
    function - 给定包含多个图像分割的一个数组，排布显示分割图像掩码。

    Paras:
        segs_array - 多个图像分割数组；ndarray
        columns - 列数；int
        titles - 子图标题；string
        figsize - 图表大小。The default is (20,10)；tuple(int)
    '''       
    
    rows=math.ceil(segs_array.shape[0]/columns)
    fig,axes=plt.subplots(rows,columns,sharex=True,sharey=True,figsize=figsize)   # 布局多个子图，每个子图显示一幅图像
    ax=axes.flatten()  # 降至1维，便于循环操作子图
    for i in range(segs_array.shape[0]):
        ax[i].imshow(segs_array[i],cmap=cmap)  # 显示图像
        ax[i].set_title("{}={}".format(prefix,titles[i]))
    invisible_num=rows*columns-len(segs_array)
    if invisible_num>0:
        for i in range(invisible_num):
            ax.flat[-(i+1)].set_visible(False)
    fig.tight_layout() # 自动调整子图参数，使之填充整个图像区域  
    fig.suptitle("segs show",fontsize=14,fontweight='bold',y=1.02)
    plt.show()    