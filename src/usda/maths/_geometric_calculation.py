# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 10:59:06 2022

@author: richie bao
"""
import math
import numpy as np

def circle_lines(center,radius,division):    
    '''
    function - 给定圆心，半径，划分份数，计算所有直径的首尾坐标

    Params:
        center - 圆心，例如(0,0)；tuple
        radius - 半径；float
        division - 划分份数；int
        
    Returns:
        xy - 首坐标数组；array
        xy_ -尾坐标数组；array
        xy_head_tail - 收尾坐标数组；array
    '''    
    
    angles=np.linspace(0,2*np.pi,division)
    x=np.cos(angles)*radius
    y=np.sin(angles)*radius
    xy=np.array(list(zip(x,y)))    
    xy=xy+center

    x_=-x
    y_=-y
    xy_=np.array(list(zip(x_,y_)))
    xy_=xy_+center
    
    xy_head_tail=np.concatenate((xy,xy_),axis=1)
    return xy,xy_,xy_head_tail


def point_Proj2Line(line_endpts,point):    
    '''
    function - 计算二维点到直线上的投影
    
    Params:
        line_endpts - 直线首尾点坐标，例如((2,0),(-2,0))；tuple/list
        point - 待要投影的点，例如[-0.11453985,  1.23781631]；tuple/list
    
    Returns:
        P - 投影点；tuple/list
    '''
    import numpy as np
    
    pts=np.array(point)
    Line_s=np.array(line_endpts[0])
    Line_e=np.array(line_endpts[1])
    
    n=Line_s - Line_e
    n_=n/np.linalg.norm(n, 2)    
    P=Line_e + n_*np.dot(pts - Line_e, n_)
    
    return P