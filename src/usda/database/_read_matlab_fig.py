# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 16:27:07 2022

@author: richie bao
"""
from scipy.io import loadmat
import matplotlib.pyplot as plt  

def read_MatLabFig_type_A(matLabFig_fp,plot=True):
    '''
    function - 读取MatLab的图表数据，类型-A
    
    Params:
        matLabFig_fp - MatLab的图表数据文件路径；string
    
    Returns:
        fig_dic - 返回图表数据，（X,Y,Z）
    '''  
    
    matlab_fig=loadmat(matLabFig_fp, squeeze_me=True, struct_as_record=False)
    fig_dic={} # 提取MatLab的.fig值
    ax1=[c for c in matlab_fig['hgS_070000'].children if c.type == 'axes']
    if(len(ax1) > 0):
        ax1 = ax1[0]
    i=0
    for line in ax1.children:
        try:
            X=line.properties.XData # good   
            Y=line.properties.YData 
            Z=line.properties.ZData
            fig_dic[i]=(X,Y,Z)
        except:
            pass     
        i+=1
        
    if plot==True:
        fig=plt.figure(figsize=(130,20))
        markers=['.','+','o','','','']
        colors=['#7f7f7f','#d62728','#1f77b4','','','']
        linewidths=[2,10,10,0,0,0]
        
        plt.plot(fig_dic[1][1],fig_dic[1][2],marker=markers[0], color=colors[0],linewidth=linewidths[0])  
        
        plt.tick_params(axis='both',labelsize=40)
        plt.show()
    
    return fig_dic