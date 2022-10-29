# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 16:17:19 2022

@author: richie bao
"""
import pandas as pd
import numpy as np
from sklearn.metrics import median_absolute_error, mean_absolute_error
import matplotlib.pyplot as plt 

from shapely.geometry import Point, LineString, shape
import geopandas as gpd
import pyproj


class movingAverage_inflection:  
    '''
    class - 曲线（数据）平滑，与寻找曲线水平和纵向的斜率变化点
    
    Params:
        series - pandas 的Series格式数据
        window - 滑动窗口大小，值越大，平滑程度越大
        plot_intervals - 是否打印置信区间，某人为False 
        scale - 偏差比例，默认为1.96, 
        plot_anomalies - 是否打印异常值，默认为False,
        figsize - 打印窗口大小，默认为(15,5),
        threshold - 拐点阈值，默认为0
    '''
    
    def __init__(self,series, window, plot_intervals=False, scale=1.96, plot_anomalies=False,figsize=(15,5),threshold=0):
        self.series=series
        self.window=window
        self.plot_intervals=plot_intervals
        self.scale=scale
        self.plot_anomalies=plot_anomalies
        self.figsize=figsize
        
        self.threshold=threshold
        self.rolling_mean=self.movingAverage()
    
    def masks(self,vec):
        '''
        function - 寻找曲线水平和纵向的斜率变化，参考 https://stackoverflow.com/questions/47342447/find-locations-on-a-curve-where-the-slope-changes
        '''
        
        d=np.diff(vec)
        dd=np.diff(d)

        # Mask of locations where graph goes to vertical or horizontal, depending on vec
        to_mask=((d[:-1] != self.threshold) & (d[:-1] == -dd-self.threshold))
        # Mask of locations where graph comes from vertical or horizontal, depending on vec
        from_mask=((d[1:] != self.threshold) & (d[1:] == dd-self.threshold))
        return to_mask, from_mask
        
    def apply_mask(self,mask, x, y):
        return x[1:-1][mask], y[1:-1][mask]   
    
    def knee_elbow(self):
        '''
        function - 返回拐点的起末位置
        '''        
        
        x_r=np.array(self.rolling_mean.index)
        y_r=np.array(self.rolling_mean)
        to_vert_mask, from_vert_mask=self.masks(x_r)
        to_horiz_mask, from_horiz_mask=self.masks(y_r)     

        to_vert_t, to_vert_v=self.apply_mask(to_vert_mask, x_r, y_r)
        from_vert_t, from_vert_v=self.apply_mask(from_vert_mask, x_r, y_r)
        to_horiz_t, to_horiz_v=self.apply_mask(to_horiz_mask, x_r, y_r)
        from_horiz_t, from_horiz_v=self.apply_mask(from_horiz_mask, x_r, y_r)    
        return x_r,y_r,to_vert_t, to_vert_v,from_vert_t, from_vert_v,to_horiz_t, to_horiz_v,from_horiz_t, from_horiz_v

    def movingAverage(self):
        rolling_mean=self.series.rolling(window=self.window).mean()        
        return rolling_mean        

    def plot_movingAverage(self,inflection=False):
        """
        function - 打印移动平衡/滑动窗口，及拐点
        """       

        plt.figure(figsize=self.figsize)
        plt.title("Moving average\n window size = {}".format(self.window))
        plt.plot(self.rolling_mean, "g", label="Rolling mean trend")

        # 打印置信区间，Plot confidence intervals for smoothed values
        if self.plot_intervals:
            mae=mean_absolute_error(self.series[self.window:], self.rolling_mean[self.window:])
            deviation=np.std(self.series[self.window:] - self.rolling_mean[self.window:])
            lower_bond=self.rolling_mean - (mae + self.scale * deviation)
            upper_bond=self.rolling_mean + (mae + self.scale * deviation)
            plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
            plt.plot(lower_bond, "r--")

            # 显示异常值，Having the intervals, find abnormal values
            if self.plot_anomalies:
                anomalies=pd.DataFrame(index=self.series.index, columns=self.series.to_frame().columns)
                anomalies[self.series<lower_bond]=self.series[self.series<lower_bond].to_frame()
                anomalies[self.series>upper_bond]=self.series[self.series>upper_bond].to_frame()
                plt.plot(anomalies, "ro", markersize=10)
                
        if inflection:
            x_r,y_r,to_vert_t, to_vert_v,from_vert_t, from_vert_v,to_horiz_t, to_horiz_v,from_horiz_t, from_horiz_v=self.knee_elbow()
            plt.plot(x_r, y_r, 'b-')
            plt.plot(to_vert_t, to_vert_v, 'r^', label='Plot goes vertical')
            plt.plot(from_vert_t, from_vert_v, 'kv', label='Plot stops being vertical')
            plt.plot(to_horiz_t, to_horiz_v, 'r>', label='Plot goes horizontal')
            plt.plot(from_horiz_t, from_horiz_v, 'k<', label='Plot stops being horizontal')     
            

        plt.plot(self.series[self.window:], label="Actual values")
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.xticks(rotation='vertical')
        plt.show()

def vanishing_position_length(matches_num,coordi_df,epsg,**kwargs):
    '''
    function - 计算图像匹配特征点几乎无关联的距离，即对特定位置视觉随距离远去而感知消失的距离
    
    Params:
        matches_num - 由类dynamicStreetView_visualPerception计算的特征关键点匹配数量
        coordi_df - 包含经纬度的DataFrame，其列名为：lon,lat
        **kwargs - 同类movingAverage_inflection配置参数
    '''  
    
    MAI_paras={'window':15,'plot_intervals':True,'scale':1.96, 'plot_anomalies':True,'figsize':(15*2,5*2),'threshold':0}
    MAI_paras.update(kwargs)   
    
    vanishing_position={}
    for idx in range(len(matches_num)): 
        x=np.array(range(idx,idx+len(matches_num[idx]))) 
        y=np.array(matches_num[idx])
        y_=pd.Series(y,index=x)   
        MAI=movingAverage_inflection(y_, 
                                     window=MAI_paras['window'],
                                     plot_intervals=MAI_paras['plot_intervals'],
                                     scale=MAI_paras['scale'], 
                                     plot_anomalies=MAI_paras['plot_anomalies'],
                                     figsize=MAI_paras['figsize'],
                                     threshold=MAI_paras['threshold'])   
        _,_,_,_,from_vert_t, _,_, _,from_horiz_t,_=MAI.knee_elbow()
        if np.any(from_horiz_t!= None) :
            vanishing_position[idx]=(idx,from_horiz_t[0])
        else:
            vanishing_position[idx]=(idx,idx)
    vanishing_position_df=pd.DataFrame.from_dict(vanishing_position,orient='index',columns=['start_idx','end_idx'])

    vanishing_position_df['geometry']=vanishing_position_df.apply(lambda idx:LineString(coordi_df[idx.start_idx:idx.end_idx][['lon','lat']].apply(lambda row:Point(row.lon,row.lat), axis=1).values.tolist()), axis=1)
    crs_4326=4326
    vanishing_position_gdf=gpd.GeoDataFrame(vanishing_position_df,geometry='geometry',crs=crs_4326)
    
    crs_=pyproj.CRS(epsg) 
    vanishing_position_gdf_reproj=vanishing_position_gdf.to_crs(crs_)
    vanishing_position_gdf_reproj['length']=vanishing_position_gdf_reproj.geometry.length
    
    return vanishing_position_gdf_reproj