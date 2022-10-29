# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 08:35:25 2022

@author: richie bao
"""
from skimage import exposure
import numpy as np 

from statistics import multimode
from tqdm import tqdm  

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