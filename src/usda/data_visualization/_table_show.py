# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 08:40:17 2022

@author: richie bao
"""
import plotly.graph_objects as go
import pandas as pd 
from IPython.display import HTML

def plotly_table(df,column_extraction):
    '''
    funciton - 使用Plotly，以表格形式显示DataFrame格式数据
    
    Params:
        df - 输入的DataFrame或者GeoDataFrame；[Geo]DataFrame
        column_extraction - 提取的字段（列）；list(string)
    
    Returns:
        None
    ''' 
    import plotly.io as pio
    pio.renderers.default='browser' 

    fig=go.Figure(data=[go.Table(
        header=dict(values=column_extraction,
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=df[column_extraction].values.T.tolist(), #values参数值为按列的嵌套列表，因此需要使用参数.T反转数组
                   fill_color='lavender',
                   align='left'))
                   ])
    fig.show() 
    
def print_html(df,row_numbers=5):    
    '''
    function - 在Jupyter中打印DataFrame格式数据为HTML
    
    Params:
        df - 需要打印的DataFrame或GeoDataFrame格式数据；DataFrame
        row_numbers - 打印的行数，如果为正，从开始打印如果为负，从末尾打印；int
        
    Returns:
        转换后的HTML格式数据；
     '''    
    if row_numbers>0:
        return HTML(df.head(row_numbers).to_html())
    else:
        return HTML(df.tail(abs(row_numbers)).to_html())    
    
    