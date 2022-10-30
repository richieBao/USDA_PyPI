# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 15:09:18 2022

@author: richie bao
"""
import numpy as np
from skimage.segmentation import felzenszwalb
from tqdm import tqdm # conda install -c conda-forge tqdm ;conda install -c conda-forge ipywidgets  
from scipy.ndimage import label

def superpixel_segmentation_Felzenszwalb(img,scale_list,sigma=0.5, min_size=50):
    '''
    function - 超像素分割，skimage库felzenszwalb方法。给定scale参数列表，批量计算
    
    Params:
        img - 读取的遥感影像、图像；ndarray
        scale_list - 分割比例列表；list(float)
        sigma - Width (standard deviation) of Gaussian kernel used in preprocessing.The default is 0.5； float
        min_size - Minimum component size. Enforced using postprocessing. The default is 50； int
        
    Returns:
        分割结果。Integer mask indicating segment labels；ndarray
    '''  
    
    segments=[felzenszwalb(img, scale=s, sigma=sigma, min_size=min_size) for s in tqdm(scale_list)]
    return np.stack(segments)


def superpixel_segmentation_quickshift(img,kernel_sizes, ratio=0.5):
    '''
    function - 超像素分割，skimage库quickshift方法。给定kernel_size参数列表，批量计算
    
    Params:
        img - Input image. The axis corresponding to color channels can be specified via the channel_axis argument；ndarray
        kernel_sizes - Width of Gaussian kernel used in smoothing the sample density. Higher means fewer clusters；float, optional
        ratio - Balances color-space proximity and image-space proximity. Higher values give more weight to color-space. The default is 0.5；float, optional, between 0 and 1
        
    Returns:
        Integer mask indicating segment labels.
    '''
    import numpy as np
    from skimage.segmentation import quickshift
    from tqdm import tqdm # conda install -c conda-forge tqdm ;conda install -c conda-forge ipywidgets    
    
    segments=[quickshift(img, kernel_size=k,ratio=ratio) for k in tqdm(kernel_sizes)]
    return np.stack(segments)

def multiSegs_stackStatistics(segs,save_fp):
    '''
    function - 多尺度超像素级分割结果叠合频数统计
    
    Params:
        segs - 超级像素分割结果。Integer mask indicating segment labels；ndarray（int）
        save_fp - 保存路径名（pickle）；string
        
    Returns:
        stack_statistics - 统计结果字典；dict
    '''  
    
    segs=list(reversed(segs))
    stack_statistics={}
    for i in tqdm(range(len(segs)-1)):
        labels=np.unique(segs[i])
        coords=[np.column_stack(np.where(segs[i]==k)) for k in labels]
        i_j={}
        for j in range(i+1,len(segs)):
            j_k={}
            for k in range(len(coords)):
                covered_elements=[segs[j][x,y] for x,y in zip(*coords[k].T)]
                freq=list(zip(np.unique(covered_elements, return_counts=True)))
                j_k[k]=freq
            i_j[j]=j_k
            
        stack_statistics[i]=i_j
    with open(save_fp,'wb') as f:
        pickle.dump(stack_statistics,f)
    
    return stack_statistics