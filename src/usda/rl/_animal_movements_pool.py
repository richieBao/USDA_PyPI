# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 10:10:26 2023

@author: richie bao
"""
import  usda.utils as usda_utils
import numpy as np

def neighbors_rewards(row_col,args):
    model,array2d,r=args
    row,col=row_col
    nbr_xy=usda_utils.grid_neighbors(array2d,row,col,r=r)
    nbr_dist=usda_utils.grid_distance(nbr_xy,row,col)
    group_sum_dict=usda_utils.group_sum(array2d,nbr_xy,1/nbr_dist)  
    X_dict={1:0,4:0,5:0,6:0,7:0,2:0}
    keys=[1,4,5,6,7,2]
    X_dict.update(group_sum_dict)
    X=np.array([[X_dict[k] for k in keys]])
    y=model.predict(X)
    
    return (row,col,y[0])