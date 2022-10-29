# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 16:31:29 2022

@author: richie bao
"""
import numpy as np
from scipy import stats

def curve_segmentation_1DConvolution(data,threshold=1):
    '''
    function - 应用一维卷积，根据跳变点分割数据
    
    Params:
        data - 待处理的一维度数据；list/array
    
    Returns:
        data_seg - 列表分割字典，"dataIdx_jump"-分割索引值，"data_jump"-分割原始数据，"conv_jump"-分割卷积结果
    '''
    
    def lst_index_split(lst, args):
        '''
        function - 根据索引，分割列表
        
        transfer:https://codereview.stackexchange.com/questions/47868/splitting-a-list-by-indexes/47877 
        '''
        if args:
            args=(0,) + tuple(data+1 for data in args) + (len(lst)+1,)
        seg_list=[]
        for start, end in zip(args, args[1:]):
            seg_list.append(lst[start:end])
        return seg_list
    
    data=data.tolist()
    kernel_conv=[-1,2,-1] # 定义卷积核，即指示函数
    result_conv=np.convolve(data,kernel_conv,'same')
    # 标准化，方便确定阈值，根据阈值切分列表
    z=np.abs(stats.zscore(result_conv)) # 标准计分-绝对值
    z_=stats.zscore(result_conv) # 标准计分
    
    threshold=threshold
    breakPts=np.where(z > threshold) # 返回满足阈值的索引值
    breakPts_=np.where(z_ < -threshold)
    
    # 根据满足阈值的索引值，切分列表
    conv_jump=lst_index_split(result_conv.tolist(), breakPts_[0].tolist()) # 分割卷积结果
    data_jump=lst_index_split(data, breakPts_[0].tolist()) # 分割原始数据
    dataIdx_jump=lst_index_split(list(range(len(data))), breakPts_[0].tolist()) # 分割索引值
    data_seg={"dataIdx_jump":dataIdx_jump,"data_jump":data_jump,"conv_jump":conv_jump}
    
    return data_seg