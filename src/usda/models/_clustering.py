# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 10:19:49 2023

@author: richie bao
"""
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_selection import SelectKBest
from tqdm import tqdm
import numpy as np

def clustering_minibatchkmeans_selectkbest_ns(df,cols,ns,score_func,k='all',**kwargs):    
    args=dict(batch_size=4096,
               init="k-means++",
               n_init=10,
               max_no_improvement=10,
               verbose=0)
    args.update(kwargs)
    
    array=df[cols].to_numpy()
    
    labels_lst=[]
    best_scores_lst=[]
    for n in tqdm(ns):
        mbk=MiniBatchKMeans(n_clusters=n,**args)
        mbk.fit(array)
        labels_lst.append(mbk.labels_)
        
        select=SelectKBest(score_func=score_func, k=k)
        select.fit_transform(array,mbk.labels_)
        best_scores_lst.append(select.scores_)
    
    labels=np.array(labels_lst)
    best_scores=np.array(best_scores_lst)

    return labels,best_scores
