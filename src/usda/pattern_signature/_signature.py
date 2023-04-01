# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 10:41:13 2023

@author: richie bao
"""
import numpy as np
import pandas as pd
import math
import itertools
from collections import Counter
from ..pattern_signature import  _grid_neighbors_xy_finder as nghxy_finder
from toolz import partition

def lexsort_based(data):                 
    '''
    将2维数组按行，移除相同的行。例如[[3,1],[3,1],[2,6]]结果为[[3,1],[2,6]]

    Parameters
    ----------
    data : 2darray
        2维数组.

    Returns
    -------
    2darray
        移除相同行后的结果.

    '''
    sorted_data=data[np.lexsort(data.T),:]
    row_mask=np.append([True],np.any(np.diff(sorted_data,axis=0),1))
    return sorted_data[row_mask]

def class_clumpSize_histogram(class_2darray,clump_2darray,base=2, right=False):
    '''
    计算类/簇大小 直方图 （class/clump-size histogram），其中簇需预先由cc3d库的connected_components等方法计算

    Parameters
    ----------
    class_2darray : 2darray
        分类数组.
    clump_2darray : 2darray
        簇类数组.
    base : int, optional
        用于构建频数宽度（bins）的基数. The default is 2.

    Returns
    -------
    class_inds_frequency_df : DataFrame
        列为分类，行为簇类的频数矩阵.

    '''    
    
    unique, counts=np.unique(clump_2darray, return_counts=True)
    class_clump=np.stack((class_2darray,clump_2darray),axis=2)
    class_clump_mapping=lexsort_based(class_clump.reshape(-1,2)).tolist()
    class_clump_mapping.sort(key=lambda x:x[0])
    counts_max=max(counts)
    bins=[]
    c=itertools.count()
    
    next(c)    
    while True:
        val=math.pow(base,next(c))
        bins.append(val)
        if val>counts_max:break
        
    bins.insert(0,0)      
    inds=np.digitize(counts,bins,right=right)
    class_inds_mapping=dict(zip(unique,inds))
    clump_counts_mapping=dict(zip(unique,counts))                          
    func=lambda row:[row[0],class_inds_mapping[row[1]],clump_counts_mapping[row[1]]]
    class_inds_counts=np.apply_along_axis(func,-1,class_clump_mapping).astype(int)
    class_inds_counts_df=pd.DataFrame(class_inds_counts,columns=['class','inds','counts'])
    class_inds_counts_group=class_inds_counts_df.groupby(by=['class','inds']).sum()
    class_inds_frequency_df=class_inds_counts_group.unstack(level=0,fill_value=0)
    
    class_inds_frequency_df.columns=class_inds_frequency_df.columns.get_level_values(1)
    return class_inds_frequency_df    

def class_co_occurrence(class_2darray):
    '''
    计算分类2维数组，相邻值对（8对）的频数。分类两两组合数为（k^2+k）/2

    Parameters
    ----------
    class_2darray : 2darray
        分类2维数组.

    Returns
    -------
    class_pairs_frequency : dict
        分类两两组合频数.

    '''
    xextent,yextent=class_2darray.shape
    ngh_finder=nghxy_finder.GridNghFinder(0, 0, xextent-1,yextent-1)
    
    x_=np.linspace(0, xextent-1, xextent)
    y_=np.linspace(0, yextent-1, yextent)
    x_idx, y_idx=np.meshgrid(x_, y_)  
    xy=np.stack((x_idx,y_idx),axis=2).reshape(-1,2).astype(int)
    pairs=np.empty((0,2),int)
    for i in xy:
        nghs=ngh_finder.find(i[0],i[1])
        ngh_vals=[class_2darray[j[0],j[1]] for j in nghs]
        i_val=class_2darray[i[0],i[1]]
        ngh_vals.remove(i_val)
        i_pairs=np.array([[i_val,k] for k in ngh_vals])
        pairs=np.append(pairs,i_pairs,axis=0)
    
    
    pairs_tuple=[tuple(i) for i in pairs]
    paris_frequency=Counter(pairs_tuple)
    
    class_unique=np.unique(class_2darray)
    different_class_pairs=list(itertools.combinations(class_unique,2))
    same_class_pairs=[(i,i) for i in class_unique]    
    
    class_pairs_frequency={p:paris_frequency[p] for p in different_class_pairs}
    same_class_pairs_frequency={p:paris_frequency[p]/2 for p in same_class_pairs}
    class_pairs_frequency.update(same_class_pairs_frequency)
    
    return pd.DataFrame.from_dict(class_pairs_frequency,orient='index')

def tags_classify(x):
    if x<1/4:return 1
    elif x>1/2:return 3
    else:return 2

def class_decomposition(class_2darray):
    '''
    层级分解。对分类2维数据，按四叉树（quadtree）的方式统计各个层级分类频数百分比区间（小于1/2为1，位于1/2和1/4时为2， 大于1/2时为3）的频数
    
    参考文献：
    1. Jasiewicz, J., Netzel, P. & Stepinski, T. GeoPAT: A toolbox for pattern-based information retrieval from large geospatial databases. Comput Geosci 80, 62–73 (2015).
    2. Remmel, T. K. & Csillag, F. Mutual information spectra for comparing categorical maps. Int J Remote Sens 27, 1425–1452 (2006).

    Parameters
    ----------
    class_2darray : 2darray
        分类2维数组.

    Returns
    -------
    tags_classified_all_levels_concat : DataFrame
        层级分解，分类为列名，层级为1级索引，区间1,2,3为2级索引.

    ''' 
    array_shape=class_2darray.shape
    width_height=min(array_shape)
    D=math.floor(math.log(width_height,2))
    dims=[pow(2,L)*pow(2,L) for L in range(1,D)]
    span=[pow(2,L) for L in range(1,D)]
    idx=list(range(width_height))
    class_unique=np.unique(class_2darray)
    
    tags_classified_all_levels={}
    for i in span:        
        xy_spans=list(partition(i,idx))
        tags_classified_lst=[]
        for x in xy_spans:            
            for y in xy_spans:
                window_xy=[[x_,y_] for x_ in x for y_ in y]
                window_val=[class_2darray[xy[0],xy[1]] for xy in window_xy]
                frequency=Counter(window_val)
                class_frequency={i:0 for i in class_unique} 
                class_frequency.update(frequency)
                
                class_percentage={k:v/pow(i,2) for k,v in class_frequency.items()}
                tags_classified={k:tags_classify(v/pow(i,2)) for k,v in class_frequency.items()}
                tags_classified_lst.append(tags_classified)
        
        tags_classified_df=pd.DataFrame.from_records(tags_classified_lst)       
        tags_classified_frequency_df=pd.DataFrame({col:tags_classified_df[col].value_counts().sort_index(ascending=True).to_dict() for col in tags_classified_df.columns})
        tags_classified_all_levels[i]=tags_classified_frequency_df
   
    tags_classified_all_levels_concat=pd.concat(tags_classified_all_levels)
    
    return tags_classified_all_levels_concat.fillna(0)

def group_bins_histogram(df,cols,by,bins):
    '''
    统计给定列（属性），给定bin的频数（直方图）

    Parameters
    ----------
    df : DataFrame
        数据.
    cols : list[str]
        用于计算的列名，含用于分组的列名.
    by : str
        用于分组的列名.
    bins : list[numerical]
        列表区间，划分的宽度为(a,b].

    Returns
    -------
    cluster_bins_histogram_dict : dict[]
        按组按bin统计频数结果，键为分组值，值为DataFrame，含个区间的频数.

    '''
    
    cluster_num=df[by].value_counts().to_dict()
    cluster_bins=df[cols].groupby(by, group_keys=True).apply(lambda x:{i:pd.cut(x[i],bins).value_counts() for i in x.columns})
    cluster_bins_histogram_dict={}
    for idx,row in cluster_bins.iteritems():
        bins_fre_df=pd.DataFrame(row)
        bins_fre_df.drop(columns=[by],inplace=True)
        cluster_bins_histogram_dict[idx]=bins_fre_df/cluster_num[idx]        
        
    return cluster_bins_histogram_dict
    