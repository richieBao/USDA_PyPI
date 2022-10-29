# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 16:06:25 2022

@author: richie bao
"""
import matplotlib.pyplot as plt
from data_generator import DataGenerator
from knee_locator import KneeLocator  

def knee_lineGraph(x,y):
    '''
    function - 绘制折线图，及其拐点。需调用kneed库的KneeLocator，及DataGenerator文件

    Paras:
        x - 横坐标，用于横轴标签
        y - 纵坐标，用于计算拐点    
    ''' 
    
    # 如果调整图表样式，需调整knee_locator文件中的plot_knee（）函数相关参数
    kneedle=KneeLocator(x, y, curve='convex', direction='decreasing')
    print('曲线拐点（凸）：',round(kneedle.knee, 3))
    print('曲线拐点（凹）：',round(kneedle.elbow, 3))
    kneedle.plot_knee(figsize=(8,8))
    
    