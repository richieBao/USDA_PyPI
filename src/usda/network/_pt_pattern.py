# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 14:06:01 2023

@author: richie bao
"""
import math
import random

def closest_point(points, target):
    '''
    从一个坐标（点）列表中获得给定坐标（点）的最近点

    Parameters
    ----------
    points : list[tuple[numerical,numerical]]
        点坐标列表.
    target : tuple
        一个点坐标.

    Returns
    -------
    tuple
        最近点坐标.

    '''
    tx, ty=target
    
    return min(points, key=lambda p: (p[0] - tx)**2 + (p[1] - ty)**2)

def get_dist(a,b):
    '''
    计算两点（坐标）间的距离

    Parameters
    ----------
    a : tuple
        点坐标.
    b : tuple
        点坐标.

    Returns
    -------
    float
        距离值.

    '''
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def nni(coordinates,area):
    '''
    计算最近邻指数
    参考：https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-statistics/h-how-average-nearest-neighbor-distance-spatial-st.htm

    Parameters
    ----------
    coordinates : list[tuple[numerical,numerical]]
        点坐标列表.
    area : float
        面积.

    Returns
    -------
    nni : float
        最近邻指数.
    z_score : float
        Z_score.

    '''
    d_sum=0
    for i in range(len(coordinates)):
        coords=[coordinates[j] for j in range(len(coordinates)) if j!=i]
        nearest_neighbor=closest_point(coords,coordinates[i])
        d_sum+=get_dist(coordinates[i],nearest_neighbor)
    n=len(coordinates)
    d_o=d_sum/n
    d_e=0.5/math.sqrt(n/area)
    nni=d_o/d_e
    
    z_score=(d_o-d_e)/(0.26136/math.sqrt(pow(n,2)/area))
    
    return nni,z_score

def nni_1d(series,length=None):
    '''
    计算一维度最近邻指数

    Parameters
    ----------
    series : list[float]
        值列表.
    length : int
        参考长度.

    Returns
    -------
    flaot
        最近邻指数.
    float
        z_score.

    '''
    import math
    def closest_val(lst,target):
        return min(lst,key=lambda v:abs(v-target))
    
    d_sum=0
    for i in range(len(series)):
        series_outI=[series[j] for j in range(len(series)) if j!=i]
        neighbor_val=closest_val(series_outI,series[i])
        d_sum+=abs(series[i]-neighbor_val)
        
    if length:
        length=length
    else:
        length=max(series)
        
    n=len(series)
    d_o=d_sum/n    
    d_e=0.5/math.sqrt(n/length)
    nni=d_o/d_e
    z_score=(d_o-d_e)/(0.26136/math.sqrt(pow(n,2)/length))
    
    return nni,z_score
        
if __name__=="__main__":
    coordinates_lst=[(random.randint(0,20),random.randint(0,20))for i in range(10)]
    print(coordinates_lst)
    ann,z=nni(coordinates_lst,100)
    print(ann,z)
