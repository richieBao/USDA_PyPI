# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 14:42:26 2023

@author: richie bao
"""
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize   
from sklearn import cluster
from yellowbrick.cluster import KElbowVisualizer
import pandas as pd
import os

def connectivity4AgglomerativeClustering_pool(k,args):
    w_idx,weight_idx_dict,ptive_inf,idx=args
    if k in w_idx:
        w_v=weight_idx_dict[k]
        w_v_idx=list(w_v.keys())
        values=[w_v[i] if i in w_v_idx else ptive_inf for i in idx]
    else:
        values=[ptive_inf]*len(idx)    
    return {k:values}

def distortion_score_elbow_pool(kneighbors_graph_n_neighbors,args):
    '''
    KElbowVisualizer方法计算最优簇数，详细查看:https://www.scikit-yb.org/en/latest/api/cluster/elbow.html

    Parameters
    ----------
    idxes_df : DataFrame
        指数数据.
    fields : list
        计算的指数列名.
    save_path : string
        图表保存根目录.
    kneighbors_graph_n_neighbors : list, optional
        kneighbors_graph方法参数输入项，邻元数. The default is 9.
    k : tuple, optional
        簇数元组区间. The default is (2,12).

    Returns
    -------
    visualizer : TYPE
        包括elbow_value_（最优簇数），elbow_score_（最优簇数分值），k_scores_（所有簇数分值）.

    '''    
    idxes_df,fields,save_path,k,conn=args

    pts_geometry=idxes_df[['geometry']]    
    pts_geometry[['x','y']]=pts_geometry.geometry.apply(lambda row:pd.Series([row.x,row.y]))
    pts_coordis=pts_geometry[['x','y']].to_numpy()
    if conn is None:
        connectivity=kneighbors_graph(pts_coordis,kneighbors_graph_n_neighbors,include_self=False)           
    else:
        connectivity=conn
         
    X_=idxes_df[fields].to_numpy()
    X=normalize(X_,axis=0, norm='max')
    
    clustering_=cluster.AgglomerativeClustering(connectivity=connectivity,) 
    visualizer = KElbowVisualizer(clustering_, timings=False,size=(500, 500), k=k) #metric='calinski_harabasz'
    visualizer.fit(X)    
    # help(visualizer)
    visualizer.show(outpath=os.path.join(save_path,'KEIbow-{}.png'.format(kneighbors_graph_n_neighbors))) 
    
    return (visualizer,kneighbors_graph_n_neighbors)



