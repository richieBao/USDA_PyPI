# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 22:42:52 2023

@author: richie bao
"""
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def gdf_plot_annotate(gdf_,value_column,annotate_column=None,**setting):
    '''
    打印GeoDataFrame格式地理空间信息数据

    Parameters
    ----------
    gdf_ : GeoDataFrame
        待打印的数据.
    value_column : string
        数值显示字段名.
    annotate_column : string
        标注显示字段名.
    **setting : key args
        用于配置图表的参数，键和默认值如下
        setting_dict=dict(annotate_fontsize=8,
                          figsize=(10,10),    
                          legend_position="right",
                          legend_size="5%",
                          legend_pad=0.1,
                          legend_bbox_to_anchor=(1, 1),
                          cmap='OrRd',
                          labelsize=8,
                          scheme=None, # 等值分类图，例如 ‘BoxPlot’, ‘EqualInterval’, ‘FisherJenks’,‘FisherJenksSampled’, ‘HeadTailBreaks’, ‘JenksCaspall’, 
                                                         ‘JenksCaspallForced’, ‘JenksCaspallSampled’, ‘MaxP’, ‘MaximumBreaks’, ‘NaturalBreaks’, ‘Quantiles’, 
                                                         ‘Percentiles’, ‘StdMean’, ‘UserDefined’等
                          k=5, # 分类数量， 对应scheme参数，如果scheme参数为None，则k参数忽略
                          categorical=False # 为True时为分类数据，为False时为数值数据
                         ).

    Returns
    -------
    ax : TYPE
        DESCRIPTION.

    '''     
    
    gdf=gdf_.copy(deep=True)
    setting_dict=dict(annotate_fontsize=8,
                      figsize=(10,10),    
                      legend_position="right",
                      legend_size="5%",
                      legend_pad=0.1,
                      legend_bbox_to_anchor=(1, 1),
                      cmap='OrRd',
                      labelsize=8,
                      scheme=None,
                      k=5,
                      categorical=False
                     )
    setting_dict.update(setting)
    gdf["index"]=gdf.index
    
    fig, ax=plt.subplots(figsize=setting_dict["figsize"])
    divider=make_axes_locatable(ax) 
    if setting_dict["scheme"]:
        gdf.plot(column=value_column,scheme=setting_dict["scheme"], k= setting_dict["k"],ax=ax,legend=True,cmap=setting_dict["cmap"],legend_kwds={'bbox_to_anchor':setting_dict["legend_bbox_to_anchor"]}) 
    elif setting_dict["categorical"]:
        gdf.plot(column=value_column,categorical=True,ax=ax,legend=True,cmap=setting_dict["cmap"],edgecolor='white',legend_kwds={'bbox_to_anchor':setting_dict["legend_bbox_to_anchor"]}) 
    else:   
        cax=divider.append_axes(setting_dict["legend_position"], size=setting_dict["legend_size"], pad=setting_dict["legend_pad"]) # 配置图例参数
        gdf.plot(column=value_column,scheme=setting_dict["scheme"], k= setting_dict["k"],ax=ax,cax=cax,legend=True,cmap=setting_dict["cmap"]) 
    if annotate_column is not None:
        gdf.apply(lambda x: ax.annotate(text=x[annotate_column], xy=x.geometry.centroid.coords[0], ha='center',fontsize=setting_dict["annotate_fontsize"]),axis=1) # 增加标注
    ax.tick_params(axis='both', labelsize=setting_dict["labelsize"])

    plt.show()
