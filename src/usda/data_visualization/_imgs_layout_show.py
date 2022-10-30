# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 19:03:16 2022

@author: richie bao
"""
import math
import os
import matplotlib.pyplot as plt
from PIL import Image  

def imgs_layoutShow(imgs_root,imgsFn_lst,columns,scale,figsize=(15,10)):
    '''
    function - 显示一个文件夹下所有图片，便于查看。
    
    Params:
        imgs_root - 图像所在根目录；string
        imgsFn_lst - 图像名列表；list(string)
        columns - 列数；int
        
    Returns:
        None
    '''  
    
    rows=math.ceil(len(imgsFn_lst)/columns)
    fig,axes=plt.subplots(rows,columns,sharex=True,sharey=True,figsize=figsize)   # 布局多个子图，每个子图显示一幅图像
    ax=axes.flatten()  # 降至1维，便于循环操作子图
    for i in range(len(imgsFn_lst)):
        img_path=os.path.join(imgs_root,imgsFn_lst[i]) # 获取图像的路径
        img_array=Image.open(img_path) # 读取图像为数组，值为RGB格式0-255        
        img_resize=img_array.resize([int(scale * s) for s in img_array.size] ) # 传入图像的数组，调整图片大小
        ax[i].imshow(img_resize)  # 显示图像
        ax[i].set_title(i+1)
    fig.tight_layout() # 自动调整子图参数，使之填充整个图像区域  
    fig.suptitle("images show",fontsize=14,fontweight='bold',y=1.02)
    plt.show()
    
def imgs_layoutShow_FPList(imgs_fp_list,columns,scale,figsize=(15,10)):
    '''
    function - 显示一个文件夹下所有图片，便于查看。

    Params:
        imgs_fp_list - 图像文件路径名列表；list(string)
        columns - 显示列数；int
        scale - 调整图像大小比例因子；float
        figsize - 打印图表大小。The default is (15,10)；tuple(int)
        
    Returns:
        None
    '''

    rows=math.ceil(len(imgs_fp_list)/columns)
    fig,axes=plt.subplots(rows,columns,figsize=figsize,)   # 布局多个子图，每个子图显示一幅图像 sharex=True,sharey=True,
    ax=axes.flatten()  # 降至1维，便于循环操作子图
    for i in range(len(imgs_fp_list)):
        img_path=imgs_fp_list[i] # 获取图像的路径
        img_array=Image.open(img_path) # 读取图像为数组，值为RGB格式0-255        
        img_resize=img_array.resize([int(scale * s) for s in img_array.size] ) # 传入图像的数组，调整图片大小
        ax[i].imshow(img_resize,)  # 显示图像 aspect='auto'
        ax[i].set_title(i+1)
    invisible_num=rows*columns-len(imgs_fp_list)
    if invisible_num>0:
        for i in range(invisible_num):
            ax.flat[-(i+1)].set_visible(False)
    fig.tight_layout() # 自动调整子图参数，使之填充整个图像区域  
    fig.suptitle("images show",fontsize=14,fontweight='bold',y=1.02)
    plt.show()
        