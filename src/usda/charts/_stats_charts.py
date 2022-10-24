# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 14:05:38 2022

@author: richie bao
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def probability_graph(x_i,x_min,x_max,x_s=-9999,left=True,step=0.001,subplot_num=221,loc=0,scale=1):    
    '''
    function - 正态分布概率计算及图形表述
    
    Paras:
        x_i - 待预测概率的值；float
        x_min - 数据集区间最小值；float
        x_max - 数据集区间最大值；float
        x_s - 第2个带预测概率的值，其值大于x_i值。The default is -9999；float
        left - 是否计算小于或等于，或者大于或等于指定值的概率。The default is True；bool
        step - 数据集区间的步幅。The default is 0.001；float
        subplot_num - 打印子图的序号，例如221中，第一个2代表列，第二个2代表行，第三个是子图的序号，即总共2行2列总共4个子图，1为第一个子图。The default is 221；int
        loc - 即均值。The default is 0；float
        scale - 标准差。The default is 1；float
        
    Returns:
        None
    '''
    x=np.arange(x_min,x_max,step)
    ax=plt.subplot(subplot_num)
    ax.margins(0.2) 
    ax.plot(x,norm.pdf(x,loc=loc,scale=scale))
    ax.set_title('N(%s,$%s^2$),x=%s'%(loc,scale,x_i))
    ax.set_xlabel('x')
    ax.set_ylabel('pdf(x)')
    ax.grid(True)
    
    if x_s==-9999:
        if left:
            px=np.arange(x_min,x_i,step)
            ax.text(loc-loc/10,0.01,round(norm.cdf(x_i,loc=loc,scale=scale),3), fontsize=20)
        else:
            px=np.arange(x_i,x_max,step)
            ax.text(loc+loc/10,0.01,round(1-norm.cdf(x_i,loc=loc,scale=scale),3), fontsize=20)
        
    else:
        px=np.arange(x_s,x_i,step)
        ax.text(loc-loc/10,0.01,round(norm.cdf(x_i,loc=loc,scale=scale)-norm.cdf(x_s,loc=loc,scale=scale),2), fontsize=20)
    ax.set_ylim(0,norm.pdf(loc,loc=loc,scale=scale)+0.005)
    ax.fill_between(px,norm.pdf(px,loc=loc,scale=scale),alpha=0.5, color='g')
    
def demo_con_style(a_coordi,b_coordi,ax,connectionstyle):
    '''
    function - 在matplotlib的子图中绘制连接线。参考： matplotlib官网Connectionstyle Demo
   
    Params:
        a_coordi - a点的x，y坐标；tuple
        b_coordi - b点的x，y坐标；tuple
        ax - 子图；ax(plot)
        connectionstyle - 连接线的形式；string
        
    Returns:
        None
    '''
    x1, y1=a_coordi[0],a_coordi[1]
    x2, y2=b_coordi[0],b_coordi[1]

    ax.plot([x1, x2], [y1, y2], ".")
    ax.annotate("",
                xy=(x1, y1), xycoords='data',
                xytext=(x2, y2), textcoords='data',
                arrowprops=dict(arrowstyle="->", color="0.5",
                                shrinkA=5, shrinkB=5,
                                patchA=None, patchB=None,
                                connectionstyle=connectionstyle,
                                ),
                )

    ax.text(.05, .95, connectionstyle.replace(",", ",\n"),
            transform=ax.transAxes, ha="left", va="top")
    
    
    
    
    