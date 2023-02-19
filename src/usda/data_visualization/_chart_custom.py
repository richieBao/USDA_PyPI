# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 14:11:37 2022

@author: richie bao
"""
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib import cm
import matplotlib
import numpy as np

def boxplot_custom(data_dict,**args):
    '''
    根据matplotlib库的箱型图打印方法，自定义箱型图可调整的打印样式。 

    Parameters
    ----------
    data_dict : dict(list,numerical)
        字典结构形式的数据，键为横坐分类数据，值为数值列表.
    **args : keyword arguments
        可调整的箱型图样式参数包括['figsize',  'fontsize',  'frameOn',  'xlabel',  'ylabel',  'labelsize',  'tick_length',  'tick_width',  'tick_color',  'tick_direction',  'notch',  'sym',  'whisker_linestyle',  'whisker_linewidth',  'median_linewidth',  'median_capstyle'].

    Returns
    -------
    paras : dict
        样式更新后的参数值.

    '''    
    
    # 计算值提取
    data_keys=list(data_dict.keys())
    data_values=list(data_dict.values())     
    
    # 配置与更新参数
    paras={'figsize':(10,10),
           'fontsize':15,
           'frameOn':['top','right','bottom','left'],
           'xlabel':None,
           'ylabel':None,
           'labelsize':15,
           'tick_length':7,
           'tick_width':3,
           'tick_color':'b',
           'tick_direction':'in',
           'notch':0,
           'sym':'b+',
           'whisker_linestyle':None,
           'whisker_linewidth':None,
           'median_linewidth':None,
           'median_capstyle':'butt'}
    
    print(paras)
    paras.update(args)
    print(paras)
    
    # 根据参数调整打印图表样式
    plt.rcParams.update({'font.size': paras['fontsize']})
    frameOff=set(['top','right','bottom','left'])-set(paras['frameOn'])
   
 
    # 图表打印
    fig, ax=plt.subplots(figsize=paras['figsize'])
    ax.boxplot(data_values,
               notch=paras['notch'],
               sym=paras['sym'],
               whiskerprops=dict(linestyle=paras['whisker_linestyle'],linewidth=paras['whisker_linewidth']),
               medianprops={"linewidth": paras['median_linewidth'],"solid_capstyle": paras['median_capstyle']})
    
    ax.set_xticklabels(data_keys) # 配置X轴刻度标签
    for f in frameOff:
        ax.spines[f].set_visible(False) # 配置边框是否显示
    
    # 配置X和Y轴标签
    ax.set_xlabel(paras['xlabel'])
    ax.set_ylabel(paras['ylabel'])
    
    # 配置X和Y轴标签字体大小
    ax.xaxis.label.set_size(paras['labelsize'])
    ax.yaxis.label.set_size(paras['labelsize'])
    
    # 配置轴刻度样式
    ax.tick_params(length=paras['tick_length'],
                   width=paras['tick_width'],
                   color=paras['tick_color'],
                   direction=paras['tick_direction'])

    plt.show()    
    return paras

def histogram_3d(df,**kwargs):
    '''
    打印3维柱状图（直方图）

    Parameters
    ----------
    df : DataFrame
        待打印的矩阵（数组）.
    **kwargs : kwargs
        图表样式参数，默认为：
        args=dict(dx=.3, # thickness of the bars
                  dy=.3, # thickness of the bars
                  figsize=(10,10),
                  font_size=10,
                  xlabel='X',
                  ylabel='Y',
                  zlabel='Z',
                  pitch=45,
                  roll=45,
                  zoom=0.9,).

    Returns
    -------
    None.

    '''
    # style.use('ggplot')
    
    args=dict(dx=.3,
              dy=.3,
              figsize=(10,10),
              font_size=10,
              xlabel='X',
              ylabel='Y',
              zlabel='Z',
              pitch=45,
              roll=45,
              zoom=0.9,)
    
    args.update(kwargs)    
    matplotlib.rcParams.update({'font.size': args['font_size']})
    
    # thickness of the bars
    dx, dy=args['dx'], args['dy']
    
    # prepare 3d axes
    fig=plt.figure(figsize=args['figsize'])
    ax=fig.add_subplot(111, projection='3d')    
    
    # set up positions for the bars 
    xpos=np.arange(df.shape[0])
    ypos=np.arange(df.shape[1])    
    
    
    # set the ticks in the middle of the bars
    ax.set_xticks(xpos + dx/2)
    ax.set_yticks(ypos + dy/2)    
    
    # create meshgrid 
    # print xpos before and after this block if not clear
    xpos, ypos=np.meshgrid(xpos, ypos)
    xpos=xpos.flatten()
    ypos=ypos.flatten()    
    
    # the bars starts from 0 attitude
    zpos=np.zeros(df.shape).flatten()

    # the bars' heights
    dz=df.T.values.ravel()
    colors=cm.rainbow(np.linspace(0.2, 1., xpos.ravel().shape[0]))    
    
    # plot 
    ax.bar3d(xpos,ypos,zpos,dx,dy,dz,color=colors)

    # put the column / index labels
    ax.yaxis.set_ticklabels(df.columns)
    ax.xaxis.set_ticklabels(df.index)    
    
    # name the axes
    ax.set_xlabel(args['xlabel'])
    ax.set_ylabel(args['ylabel'])
    ax.set_zlabel(args['zlabel'])

    ax.view_init(args['pitch'],args['roll']) 
    ax.set_box_aspect(aspect=None, zoom=args['zoom'])
    # plt.tight_layout()
    plt.show()        
    