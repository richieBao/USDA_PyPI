# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 09:12:51 2022

@author: richie bao
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ipywidgets import widgets
from IPython.display import display 

def percentile_slider(season_dic):    
    '''
    function - 多个栅格数据，给定百分比，变化观察
    
    Params:
        season_dic -  多个栅格字典，键为自定义键名，值为读取的栅格数据（array），例如{"w_180310":w_180310_NDVI_rescaled,"s_190820":s_190820_NDVI_rescaled,"a_191018":a_191018_NDVI_rescaled}；dict
        
    Returns:
        None
    '''   
    
    p_1_slider=widgets.IntSlider(min=0, max=100, value=10, step=1, description="percentile_1")
    p_2_slider=widgets.IntSlider(min=0, max=100, value=30, step=1, description="percentile_2")
    p_3_slider=widgets.IntSlider(min=0, max=100, value=50, step=1, description="percentile_3")
    
    season_keys=list(season_dic.keys())
    season=widgets.Dropdown(
        description='season',
        value=season_keys[0],
        options=season_keys
    )

    season_val=season_dic[season_keys[0]]
    _,img=data_division(season_val,division=[10,30,50],right=True)
    trace1=go.Image(z=img)

    g=go.FigureWidget(data=[trace1,],
                      layout=go.Layout(
                      title=dict(
                      text='NDVI interpretation'
                            ),
                      width=800,
                      height=800
                        ))

    def validate():
        if season.value in season_keys:
            return True
        else:
            return False

    def response(change):
        if validate():
            division=[p_1_slider.value,p_2_slider.value,p_3_slider.value]
            _,img_=data_division(season_dic[season.value],division,right=True)
            with g.batch_update():
                g.data[0].z=img_
    p_1_slider.observe(response, names="value")            
    p_2_slider.observe(response, names="value")    
    p_3_slider.observe(response, names="value")    
    season.observe(response, names="value")    

    container=widgets.HBox([p_1_slider,p_2_slider,p_3_slider,season])
    box=widgets.VBox([container,g])
    display(box)