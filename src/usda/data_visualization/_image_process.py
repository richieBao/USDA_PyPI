# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 08:35:25 2022

@author: richie bao
"""
from skimage import exposure
import numpy as np 

from statistics import multimode
from tqdm import tqdm  
from PIL import Image

from pathlib import Path
import cv2 as cv
import os
import pandas as pd   


def image_exposure(img_bands,percentile=(2,98)):
    '''
    function - 拉伸图像 contract stretching
    
    Params:
        img_bands - landsat stack后的波段；array
        percentile - 百分位数，The default is (2,98)；tuple
    
    Returns:
        img_bands_exposure - 返回拉伸后的影像；array
    '''
       
    bands_temp=[]
    for band in img_bands:
        p2, p98=np.percentile(band, (2, 98))
        bands_temp.append(exposure.rescale_intensity(band, in_range=(p2,p98)))
    img_bands_exposure=np.concatenate([np.expand_dims(b,axis=0) for b in bands_temp],axis=0)
    print("exposure finished.")
    return img_bands_exposure


def downsampling_blockFreqency(array_2d,blocksize=[10,10]):
    '''
    fuction - 降采样二维数组，根据每一block内值得频数最大值，即最多出现得值为每一block的采样值
    
    Params:
        array_2d - 待降采样的二维数组；array(2d)
        blocksize - block大小，即每一采用的范围，The default is [10,10]；tuple
        
    Returns:
        downsample - 降采样结果；array 
    '''
    
    shape=array_2d.shape
    row,col=blocksize
    row_p,row_overlap=divmod(shape[1],row)  # divmod(a,b)方法为除法取整，以及a对b的余数
    col_p,col_overlap=divmod(shape[0],col)
    print("row_num:",row_p,"col_num:",col_p)
    array_extraction=array_2d[:col_p*col,:row_p*row]  # 移除多余部分，规范数组，使其正好切分均匀
    print("array extraction shape:",array_extraction.shape,"original array shape:",array_2d.shape)  
    
    h_splitArray=np.hsplit(array_extraction,row_p)
    
    v_splitArray=[np.vsplit(subArray,col_p) for subArray in h_splitArray]
    blockFrenq_list=[]
    for h in tqdm(v_splitArray):
        temp=[]
        for b in h:
            blockFrenq=multimode(b.flatten())[0]
            temp.append(blockFrenq)
        blockFrenq_list.append(temp)   
    downsample=np.array(blockFrenq_list).swapaxes(0,1)
    
    return downsample 

def img_rescale(img_path,scale):
    '''
    function - 读取与压缩图像，返回2维度数组
    
    Params:
        img_path - 待处理图像路径；lstring
        scale - 图像缩放比例因子；float
    
    Returns:
        img_3d - 返回三维图像数组；ndarray        
        img_2d - 返回二维图像数组；ndarray
    '''

    img=Image.open(img_path) # 读取图像为数组，值为RGB格式0-255  
    img_resize=img.resize([int(scale * s) for s in img.size] ) # 传入图像的数组，调整图片大小
    img_3d=np.array(img_resize)
    h, w, d=img_3d.shape
    img_2d=np.reshape(img_3d, (h*w, d))  # 调整数组形状为2维

    return img_3d,img_2d

def imgs_compression_cv(imgs_root,imwrite_root,imgsPath_fp,gap=1,png_compression=9,jpg_quality=100):
    '''
    function - 使用OpenCV的方法压缩保存图像    
    
    Params:
        imgs_root - 待处理的图像文件根目录；string
        imwrite_root - 图像保存根目录；string
        gap - 无人驾驶场景下的图像通常是紧密连续的，可以剔除部分图像避免干扰， 默认值为1；int
        png_compression - png格式压缩值，默认为9；int
        jpg_quality - jpg格式压缩至，默认为100。for jpeg only. 0 - 100 (higher means better)；int
        png_compression: For png only. 0 - 9 (higher means a smaller size and longer compression time):int
        
    Returns:
        imgs_save_fp - 保存压缩图像文件路径列表；list(string)
    ''' 
    
    if not os.path.exists(imwrite_root):
        os.makedirs(imwrite_root)
    
    imgs_root=Path(imgs_root)
    imgs_fp=[p for p in imgs_root.iterdir()][::gap]
    imgs_save_fp=[]
    for img_fp in tqdm(imgs_fp):
        img_save_fp=str(Path(imwrite_root).joinpath(img_fp.name))
        img=cv.imread(str(img_fp ))
        if img_fp.suffix=='.png':
            cv.imwrite(img_save_fp,img,[int(cv.IMWRITE_PNG_COMPRESSION), png_compression])
            imgs_save_fp.append(img_save_fp)
        elif img_fp.suffix=='.jpg':
            cv.imwrite(img_save_fp,img,[int(cv.IMWRITE_JPEG_QUALITY), jpg_quality])
            imgs_save_fp.append(strimg_save_fp)
        else:
            print("Only .jpg and .png format files are supported.")
   
    pd.DataFrame(imgs_save_fp,columns=['imgs_fp']).to_pickle(imgsPath_fp)
    return imgs_save_fp
