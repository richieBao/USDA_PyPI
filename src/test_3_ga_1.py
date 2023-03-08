# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 15:33:52 2023

@author: richie bao
"""
from usda import datasets as usda_datasets
from usda import data_visualization as usda_vis

import mapclassify







if __name__=="__main__":
    size=16
    X,_=usda_datasets.generate_categorical_2darray(size=size,seed=7)
    X4=mapclassify.FisherJenks(X[0], k=4).yb.reshape(size,size)+1
    usda_vis.imshow_label2darray(X4,figsize=(7,7),random_seed=29,fontsize=10)     
    
    
    
