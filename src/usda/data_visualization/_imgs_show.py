# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 18:30:52 2022

@author: richie bao
"""
from skimage import io
from skimage.morphology import square,rectangle
import matplotlib.pyplot as plt  
import matplotlib

import pandas as pd
import plotly.graph_objects as go 
import numpy as np

def img_struc_show(img_fp,val='R',figsize=(7,7)):
    '''
    function - 显示图像以及颜色R值，或G,B值
    
    Params:
        img_fp - 输入图像文件路径；string
        val - 选择显示值，R，G，或B，The default is 'R'；string
        figsize - 配置图像显示大小，The default is (7,7)；tuple
        
    Returns:
        None
    '''  
    
    img=io.imread(img_fp)
    shape=img.shape
    struc_square=rectangle(shape[0],shape[1])
    fig, ax=plt.subplots(figsize=figsize)
    ax.imshow(img,cmap="Paired", vmin=0, vmax=12)                     
    for i in range(struc_square.shape[0]):
        for j in range(struc_square.shape[1]):
            if val=='R':
                ax.text(j, i, img[:,:,0][i,j], ha="center", va="center", color="w")
            elif val=='G':
                ax.text(j, i, img[:,:,1][i,j], ha="center", va="center", color="w")                         
            elif val=='B':
                ax.text(j, i, img[:,:,2][i,j], ha="center", va="center", color="w")                         
    ax.set_title('structuring img elements')
    plt.show
    
def plotly_scatterMapbox(df,**kwargs):
    '''
    function - 使用plotly的go.Scattermapbox方法，在地图上显示点及其连线，坐标为经纬度
    
    Paras:
        df - DataFrame格式数据，含经纬度；DataFrame
        field - 'lon':df的longitude列名，'lat'：为df的latitude列名，'center_lon':地图显示中心精经度定位，"center_lat":地图显示中心维度定位，"zoom"：为地图缩放；string
    '''   
    
    field={'lon':'lon','lat':'lat',"center_lon":8.398104,"center_lat":49.008645,"zoom":16}
    field.update(kwargs) 
    
    fig=go.Figure(go.Scattermapbox(mode="markers",lat=df[field['lat']], lon=df[field['lon']],marker={'size': 10}))  #亦可以选择列，通过size=""配置增加显示信息 ;mode="markers+lines"
    fig.update_layout(
        margin ={'l':0,'t':0,'b':0,'r':0},
        mapbox = {
            'center': {'lon': 10, 'lat': 10},
            'style': "stamen-terrain",
            'center': {'lon': field['center_lon'], 'lat':field['center_lat']},
            'zoom': 16})    
    fig.show()    
    
def imshow_label2darray(array_2d,label=True,**kwargs):
    '''
    打印2维数组为栅格形式，并标注每个单元（像素位置）的值；cmap可配置为随机颜色

    Parameters
    ----------
    array_2d : 2darray
        2维数组.
    label : bool, optional
        是否标注单元值. The default is True.
    **kwargs : kwargs
        图表打印参数，默认值有：
        args=dict(figsize=(10,10),
                  cmap='Accent',
                  text_color='white',
                  dtype='int',
                  random_seed=30).

    Returns
    -------
    None.

    '''    
    args=dict(figsize=(10,10),
              cmap='Accent',
              text_color='white',
              dtype='int',
              random_seed=None,
              fontsize=5,
              norm=None,
              )    
    args.update(kwargs)
    
    if args['random_seed']:
        np.random.seed(args['random_seed'])
        cmap=matplotlib.colors.ListedColormap (np.random.rand(256,3))
    else:
        cmap=args['cmap']        
    
    fig,ax=plt.subplots(1,1,figsize=args['figsize']) 
    ax.imshow(array_2d,cmap=cmap,norm=args['norm'])
    
    if label:
        nx,ny=array_2d.shape
        x=np.linspace(0, nx-1, nx)
        y=np.linspace(0, ny-1, ny)
        xv, yv=np.meshgrid(x, y)   
        
        xyv=np.stack((xv,yv,array_2d),axis=2).reshape(-1,3).astype(args['dtype'])
        for x,y,i in xyv: 
            ax.text(x=x ,y=y, s=i,ha='center', va='center',color=args['text_color'],fontsize=args['fontsize'])    
    plt.show()