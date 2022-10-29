# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 08:59:28 2022

@author: richie bao
"""
import numpy as np

def k_neighbors_entire(xy,k=3):    
    '''
    function - 返回指定邻近数目的最近点坐标
    
    Params:
        xy - 点坐标二维数组，例如
            array([[23714. ,   364.7],
                  [21375. ,   331.4],
                  [32355. ,   353.7],
                  [35503. ,   273.3]]
        k - 指定邻近数目；int
    
    return:
        neighbors - 返回各个点索引，以及各个点所有指定数目邻近点索引；list(tuple)
    '''
    import numpy as np
    
    neighbors=[(s,np.sqrt(np.sum((xy-xy[s])**2,axis=1)).argsort()[1:k+1]) for s in range(xy.shape[0])]
    return neighbors


if __name__=="__main__":
    xy= np.array([[23714. ,   364.7],
                  [21375. ,   331.4],
                  [32355. ,   353.7],
                  [35503. ,   273.3]])
    neighbors=k_neighbors_entire(xy,k=3)
    print(neighbors)