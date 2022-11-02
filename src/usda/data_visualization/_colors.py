# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 08:49:11 2022

@author: richie bao
"""
import matplotlib
import numpy as np
import pandas as pd 

def generate_colors():    
    '''
    function - 生成颜色列表或者字典
    
    Returns:
        hex_colors_only - 16进制颜色值列表；list
        hex_colors_dic - 颜色名称：16进制颜色值；dict
        rgb_colors_dic - 颜色名称：(r,g,b)；dict
    '''
    hex_colors_dic={}
    rgb_colors_dic={}
    hex_colors_only=[]
    for name, hex in matplotlib.colors.cnames.items():
        hex_colors_only.append(hex)
        hex_colors_dic[name]=hex
        rgb_colors_dic[name]=matplotlib.colors.to_rgb(hex)
    return hex_colors_only,hex_colors_dic,rgb_colors_dic


def data_division(data,division,right=True):
    '''
    function - 将数据按照给定的百分数划分，并给定固定的值，整数值或RGB色彩值
    
    Params:
        data - 待划分的numpy数组；array
        division - 百分数列表，例如[0,35,85]；list
        right - bool,optional. The default is True. 
                Indicating whether the intervals include the right or the left bin edge. 
                Default behavior is (right==False) indicating that the interval does not include the right edge. 
                The left bin end is open in this case, i.e., bins[i-1] <= x < bins[i] is the default behavior for monotonically increasing bins.        
    
    Returns：
        data_digitize - 返回整数值；list(int)
        data_rgb - 返回RGB，颜色值；list
    '''   
    
    percentile=np.percentile(data,np.array(division))
    data_digitize=np.digitize(data,percentile,right)
    
    unique_digitize=np.unique(data_digitize)
    random_color_dict=[{k:np.random.randint(low=0,high=255,size=1)for k in unique_digitize} for i in range(3)]
    data_color=[pd.DataFrame(data_digitize).replace(random_color_dict[i]).to_numpy() for i in range(3)]
    data_rgb=np.concatenate([np.expand_dims(i,axis=-1) for i in data_color],axis=-1)
    
    return data_digitize,data_rgb

def uniqueish_color():
    '''
    function - 使用matplotlib提供的方法随机返回浮点型RGB
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    
    return plt.cm.gist_ncar(np.random.random())