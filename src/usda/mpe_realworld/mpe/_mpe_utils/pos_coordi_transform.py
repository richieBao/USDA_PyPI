# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 08:35:31 2023

@author: richie bao
"""
import numpy as np
from collections import defaultdict
from functools import partial

def p_pos2plat_coordi(p_pos,width,height):
    x,y=p_pos
    y *= (-1)      
    x =  x  *  height// 2
    y = y *  width// 2   
    x += height // 2
    y +=  width// 2    
    return int(x),int(y)

def plat_coordi2p_pos(plat_coordi,width,height):
    x,y=plat_coordi
    x-=height//2 
    y-=width//2
    x/=height//2
    y/=width//2 
    y*=(-1)
    return x,y   

def array2d_idxes(array2d,reverse=False):
    nx,ny=array2d.shape
    x = np.linspace(0, nx-1, nx)
    y = np.linspace(0, ny-1, ny)        
    if reverse:
        xv, yv = np.meshgrid(y, x)
        xy=np.stack((yv,xv),axis=2)
    else:
        xv, yv = np.meshgrid(x, y)
        xy=np.stack((xv,yv),axis=2)
    
    return xy

def list_duplicates(seq):
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return ((key,locs) for key,locs in tally.items() if len(locs)>0)

def list_duplicates_of(seq,item):
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item,start_at+1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs

def plat_region(array2d):
    array1d=array2d.flatten()
    region_idxes=list_duplicates(array1d)
    region_idxes=dict(region_idxes)
    return region_idxes



if __name__=="__main__":
    p_pos=[0.1397943,-0.33386614]
    p_pos=[-0.155,0.665]
    x,y=p_pos2plat_coordi(p_pos,700,300)
    print(x,y)
    
    plat_coordi=[x,y]#[170,466]
    x,y=plat_coordi2p_pos(plat_coordi,700,300)
    print(x,y)
    
    
    from usda import datasets as usda_datasets
    import mapclassify
    
    size=700
    X_,_=usda_datasets.generate_categorical_2darray(size=size,sigma=20,seed=57)
    X=X_[0].reshape(size,size)*size
    X_BoxPlot=mapclassify.BoxPlot(X)
    y=X_BoxPlot.yb.reshape(size,size)
    y=y[:300,:]
    
    xy=array2d_idxes(y,reverse=True)
    print(xy.shape)
    
    plat_region(y)