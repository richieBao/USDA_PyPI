# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 14:12:50 2023

@author: richie bao
"""
from sklearn.neighbors import kneighbors_graph
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from collections import ChainMap
import pickle 

from sklearn.preprocessing import normalize   
from sklearn import cluster
from yellowbrick.cluster import KElbowVisualizer

import matplotlib.pyplot as plt   
from pylab import mpl

import pandas as pd
from sklearn.neighbors import NearestNeighbors
import geopandas as gpd
import pyproj

from sklearn.feature_selection import SelectKBest,f_classif
import os

if __package__:
    from ._metrics_clustering_pool import connectivity4AgglomerativeClustering_pool
else:
    from _metrics_clustering_pool import connectivity4AgglomerativeClustering_pool


def connectivity4AgglomerativeClustering(idxes_df,weight_idx_dict,save_path,cpu_num=8):
    ptive_inf=float('inf')
    
    # pts_geometry=idxes_df[['geometry']]    
    # pts_geometry[['x','y']]=pts_geometry.geometry.apply(lambda row:pd.Series([row.x,row.y]))
    # pts_coordis=pts_geometry[['x','y']].to_numpy()    
    # connectivity=kneighbors_graph(pts_coordis,9,include_self=False)   
    w_idx=list(weight_idx_dict.keys())
    idx=list(range(len(idxes_df)))
    args=partial(connectivity4AgglomerativeClustering_pool, args=[w_idx,weight_idx_dict,ptive_inf,idx])
    with Pool(cpu_num) as p:
        k_values=p.map(args, tqdm(idx))      

    connectivity_dict=dict(ChainMap(*k_values))
    connectivity=np.array([connectivity_dict[k] for k in idx])
    with open(save_path,'wb') as f: 
        pickle.dump(connectivity,f)    
    print("connectivity dumped.")
    return connectivity

def distortion_score_elbow(idxes_df,fields,save_path,kneighbors_graph_n_neighbors=9,k=(2,12),conn=None):
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
    pts_geometry=idxes_df[['geometry']]    
    pts_geometry[['x','y']]=pts_geometry.geometry.apply(lambda row:pd.Series([row.x,row.y]))
    pts_coordis=pts_geometry[['x','y']].to_numpy()
    
    if conn is None:
        connectivity=kneighbors_graph(pts_coordis,kneighbors_graph_n_neighbors,include_self=False)           
    else:
        connectivity=conn    
    
    # connectivity=kneighbors_graph(pts_coordis,kneighbors_graph_n_neighbors,include_self=False)    
    X_=idxes_df[fields].to_numpy()
    X=normalize(X_,axis=0, norm='max')
    
    clustering_=cluster.AgglomerativeClustering(connectivity=connectivity,) 
    visualizer = KElbowVisualizer(clustering_, timings=False,size=(500, 500), k=k) #metric='calinski_harabasz'
    visualizer.fit(X)    
    # help(visualizer)
    visualizer.show(outpath=os.path.join(save_path,'KEIbow-{}.png'.format(kneighbors_graph_n_neighbors))) 
    
    return visualizer

def distortion_score_elbow_kneighbors(idxes_df,fields,save_path,kneighbors_graph_n_neighbors_list,k,save_fn=None): 
    '''
    跟定邻元数列表，批量计算最优簇数

    Parameters
    ----------
    idxes_df : DataFrame
        指数数据.
    fields : list
        计算的指数列名.
    save_path : string
        图表保存根目录.
    kneighbors_graph_n_neighbors_list : list
        kneighbors_graph方法参数输入项，邻元数.
    save_fn : string
        最优簇数保存路径名.
    k : tuple
        簇数元组区间.

    Returns
    -------
    elbow_score_dict : pickle(dict)
        最优簇数计算结果信息，包括k_scores，k_scores和elbow_score.

    '''    
    elbow_score_dict={}
    j=0
    for i in tqdm(kneighbors_graph_n_neighbors_list):    
        visualizer=distortion_score_elbow(idxes_df,fields,save_path,i,k)      
        elbow_score_dict[i]={'k_scores':visualizer.k_scores_,
                             'elbow_value':visualizer.elbow_value_,
                             'elbow_score':visualizer.elbow_score_} 
        # if j==0:break
        # j+=1
    print(visualizer.k_scores_,visualizer.elbow_value_,visualizer.elbow_score_ )
    if save_fn:
        with open(save_fn,'wb') as f: 
            pickle.dump(elbow_score_dict,f)     
    else:        
        return elbow_score_dict
   
def elbow_score_plot(elbow_score_dict,k,save_fn=None,figsize=(12, 14),ylim=None,legend_loc='lower left'): 
    '''
    不同邻里尺度，最优簇数图表打印

    Parameters
    ----------
    elbow_score_dict : dicit
        最优簇数计算结果信息,包括k_scores，k_scores和elbow_score.
    save_fn : string
        图表保存路径名.
    k : tuple
        簇数元组区间.

    Returns
    -------
    None.

    '''    
    mpl.rcParams['font.sans-serif']=['DengXian'] #解决中文字符乱码问题
    
    kneighbors=list(elbow_score_dict.keys())
    n_clusters=list(range(k[0],k[1]))
    k_scores=np.array([elbow_score_dict[k]['k_scores'] for k in kneighbors])
    # print(k_scores)
    y_min=k_scores.min()-100
    y_max=k_scores.max()+100
    
    fig, ax=plt.subplots(1, 1, figsize=figsize)
    # These are the colors that will be used in the plot
    ax.set_prop_cycle(color=[
        '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a',
        '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
        '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d',
        '#17becf', '#9edae5'])    
    # Remove the plot frame lines. They are unnecessary here.
    # ax.spines[:].set_visible(False)  
    for i in ['top','right',]:#'bottom','left'
        ax.spines[i].set_visible(False)
        
    for i in ['bottom','left',]:#'bottom','left'
        ax.spines[i].set_color('gray')
    
    # Ensure that the axis ticks only show up on the bottom and left of the plot.
    # Ticks on the right and top of the plot are generally unnecessary.
    # ax.xaxis.tick_bottom()
    # ax.yaxis.tick_left()    
    
    fig.subplots_adjust(left=.06, right=.75, bottom=.02, top=.94)
    # Limit the range of the plot to only where the data is.
    # Avoid unnecessary whitespace.
    # ax.set_xlim(min(n_clusters),max(n_clusters))
    ax.set_xlim(min(n_clusters),max(n_clusters))
    ax.set_ylim(y_min,y_max)    
    
    # Provide tick lines across the plot to help your viewers trace along
    # the axis ticks. Make sure that the lines are light and small so they
    # don't obscure the primary data lines.
    # ax.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)   
            
    # Remove the tick marks; they are unnecessary with the tick lines we just
    # plotted. Make sure your axis ticks are large enough to be easily read.
    # You don't want your viewers squinting to read your plot.
    # ax.tick_params(axis='both', which='both', labelsize=14,
    #                 bottom=False, top=False, labelbottom=True,
    #                 left=False, right=False, labelleft=True,)    
            
    # Now that the plot is prepared, it's time to actually plot the data!
    # Note that I plotted the majors in order of the highest % in the final year.  
    # info={kn:"kneighbors={};elbow at k={},score={}".format(kn,elbow_score_dict[kn]['elbow_value'],round(elbow_score_dict[kn]['elbow_score'],3)) for kn in kneighbors}    
    # info={kn:"邻元数={};最佳簇数 k={},分值={}".format(kn,elbow_score_dict[kn]['elbow_value'],round(elbow_score_dict[kn]['elbow_score'],3)) for kn in kneighbors}    
    # print(info)   
    info={kn:"{};{};{}".format(kn,elbow_score_dict[kn]['elbow_value'],round(elbow_score_dict[kn]['elbow_score'],3)) for kn in kneighbors}    

    y_offsets={70:15,140:-60,150:-45,130:-35,110:-25,120:-15,80:-3}    
    for kn in kneighbors:
        # Plot each line separately with its own color.    
        line,=ax.plot(n_clusters,elbow_score_dict[kn]['k_scores'] ,lw=2.5,label=info[kn])   
        #print(elbow_score_dict[kn]['elbow_value'],elbow_score_dict[kn]['elbow_score'])
        ax.scatter(elbow_score_dict[kn]['elbow_value'],elbow_score_dict[kn]['elbow_score'],marker='X',s=100)
        y_pos=elbow_score_dict[kn]['k_scores'][-1]
        # if kn in y_offsets:
        #     y_pos+=y_offsets[kn]
        # ax.text(n_clusters[-1]+0.1,y_pos,info[kn] ,fontsize=14, color=line.get_color())
        
    # ax.set_ylabel('Scores',fontsize=15)    
    # ax.set_xlabel('Number of clusters',fontsize=15)
    ax.set_ylabel('分值',fontsize=15)    
    ax.set_xlabel('簇数',fontsize=15)    
    ax.grid(False)
    
    ax.tick_params('both', length=5, width=1, which='major',color='gray',direction='in')  
    
    if ylim:
        plt.ylim(*ylim)
    plt.legend(loc=legend_loc,frameon=False)
    plt.show()
    
    if save_fn:
        fig.savefig(save_fn,bbox_inches="tight",dpi=300)

def idxes_clustering(idxes_df,fields,n_clusters=10,epsg=4326,kneighbors_graph_n_neighbors=9):
    '''
    多个指数（字段）的聚类

    Parameters
    ----------
    idxes_df : DataFrame
        指数.
    fields : list
        用于指数计算的列名列表.
    n_clusters : int, optional
        聚类数量. The default is 10.
    epsg : string or int, optional
        坐标投影系统，epsg编号. The default is 4326.

    Returns
    -------
    idxes_df_gdf : GeoDataFrame
        多个指数（字段）的聚类.

    '''    
    pts_geometry=idxes_df[['geometry']]    
    pts_geometry[['x','y']]=pts_geometry.geometry.apply(lambda row:pd.Series([row.x,row.y]))
    pts_coordis=pts_geometry[['x','y']].to_numpy()

    connectivity=kneighbors_graph(pts_coordis,kneighbors_graph_n_neighbors,include_self=False)    
    
    X_=idxes_df[fields].to_numpy()
    X=normalize(X_,axis=0, norm='max')
    clustering=cluster.AgglomerativeClustering(connectivity=connectivity,n_clusters=n_clusters).fit(X)
    cluster_column_name='clustering_{}_{}'.format(kneighbors_graph_n_neighbors,n_clusters)
    idxes_df[cluster_column_name]=clustering.labels_
    idxes_df[cluster_column_name+"_"]=idxes_df[cluster_column_name].apply(lambda row:row+1)
    
    wgs84=pyproj.CRS('EPSG:4326')
    idxes_df_gdf=gpd.GeoDataFrame(idxes_df,geometry=idxes_df.geometry,crs=wgs84)   
    idxes_df_gdf=idxes_df_gdf.to_crs(epsg)
    
    return idxes_df_gdf,cluster_column_name

def idxes_clustering_kneighbors(idxes_df,fields,kneighbors_ncluster,epsg=4326,conn=None):
    '''
    不同邻里尺度（邻元数）指数贡献度计算

    Parameters
    ----------
    idxes_df : DataFrame
        指数.
    fields : list
        用于指数计算的列名列表.
    kneighbors_ncluster : dict
        邻元数:簇数对应字典.
    epsg : int, optional
        投影坐标系统epsg编号. The default is 4326.

    Returns
    -------
    kneighbors_clusters_concat_gdf : DataFrame
        指数贡献度.
    cluster_column_name_list : list
        不同邻元数指数贡献度列名.

    '''    
    kneighbors_clusters_dict={}
    cluster_column_name_list=[]
    # i=0
    for kneighbors,n_cluster in tqdm(kneighbors_ncluster.items()):
        idxes_df_gdf,cluster_column_name=idxes_clustering(idxes_df.copy(deep=True),fields,n_cluster,epsg,kneighbors,conn=conn)
        # print(idxes_df_gdf.columns)
        kneighbors_clusters_dict[cluster_column_name]=idxes_df_gdf[[cluster_column_name]]
        cluster_column_name_list.append(cluster_column_name)
        # if i==2:break
        # i+=1
        
    kneighbors_clusters_concat=pd.concat([idxes_df.copy(deep=True)]+list(kneighbors_clusters_dict.values()),axis=1) 
    wgs84=pyproj.CRS('EPSG:4326')
    kneighbors_clusters_concat_gdf=gpd.GeoDataFrame(kneighbors_clusters_concat,geometry=kneighbors_clusters_concat.geometry,crs=wgs84) 
    # print(cluster_column_name_list)
    kneighbors_clusters_concat_gdf=kneighbors_clusters_concat_gdf.to_crs(epsg)    
    return kneighbors_clusters_concat_gdf,cluster_column_name_list     
       
def idxes_clustering_contribution(idxes_df,fields,n_clusters=10,kneighbors_graph_n_neighbors=9,conn=None,show=False):
    '''
    聚类指数贡献度计算

    Parameters
    ----------
    idxes_df : DataFrame
        指数.
    fields : list
        用于指数计算的列名列表.
    n_clusters : int, optional
        聚类簇数. The default is 10.
    kneighbors_graph_n_neighbors : int, optional
        kneighbors_graph方法参数输入项，邻元数. The default is 9.

    Returns
    -------
    featureScores : DataFrame
        贡献度.

    '''
    # import matplotlib    
    # font = {
    #         # 'family' : 'normal',
    #         # 'weight' : 'bold',
    #         'size'   : 28}
    # matplotlib.rc('font', **font) 
    
    pts_geometry=idxes_df[['geometry']]    
    pts_geometry[['x','y']]=pts_geometry.geometry.apply(lambda row:pd.Series([row.x,row.y]))
    pts_coordis=pts_geometry[['x','y']].to_numpy()
    
    # nbrs=NearestNeighbors(n_neighbors=kneighbors_graph_n_neighbors, algorithm='ball_tree').fit(pts_coordis)
    # connectivity=nbrs.kneighbors_graph(pts_coordis)
    # connectivity=kneighbors_graph(pts_coordis,kneighbors_graph_n_neighbors,include_self=False)
    
    if conn is None:
        connectivity=kneighbors_graph(pts_coordis,kneighbors_graph_n_neighbors,include_self=False)
    else:
        connectivity=conn    
    
    X_=idxes_df[fields].to_numpy()
    X=normalize(X_,axis=0, norm='max')

    clustering=cluster.AgglomerativeClustering(connectivity=connectivity,n_clusters=n_clusters).fit(X)
    y=clustering.labels_
    selector=SelectKBest(score_func=f_classif, k=len(fields)) #score_func=chi2    
    selector.fit(X,y)
    
    dfscores = pd.DataFrame(selector.scores_)
    dfpvalues=pd.DataFrame(selector.pvalues_)
    dfcolumns = pd.DataFrame(fields)  
    featureScores=pd.concat([dfcolumns,dfscores,dfpvalues],axis=1)
    featureScores.columns=['Factor','Score','p_value']  #naming the dataframe columns
    # print(featureScores)
    
    featureScores_=featureScores.set_index('Factor')    
    # featureScores_.nlargest(len(fields),'Score').Score.plot(kind='barh',figsize=(30,20),fontsize=38)
    if show:
        featureScores_.Score.plot(kind='barh')
        plt.show()    

    return featureScores

def idxes_clustering_contribution_kneighbors(idxes_df,fields,elbow_score_dict,save_fn=None,conn=None):
    '''
    不同邻里尺度，聚类指数贡献度计算

    Parameters
    ----------
    idxes_df : DataFrame
        指数.
    fields : list
        用于指数计算的列名列表.
    elbow_score_dict : dict
        最优簇数计算结果信息，包括k_scores，k_scores和elbow_score.
    save_fn : string
        保存路径名.

    Returns
    -------
    featureScores_dict : dict
        聚类指数贡献度.

    '''       
    kneighbors=list(elbow_score_dict.keys())
    elbow_value={k:elbow_score_dict[k]['elbow_value'] for k in kneighbors}
    featureScores_dict={}
    for k in tqdm(kneighbors):
        featureScores=idxes_clustering_contribution(idxes_df,fields,elbow_value[k],k,conn=conn)
        featureScores_dict[k]=featureScores
        
    if save_fn:
        with open(save_fn, 'wb') as f:
            pickle.dump(featureScores_dict,f)
        
    return featureScores_dict

def idxes_clustering_contribution_kneighbors_plot_(featureScores_dict,save_fn=None,figsize=(12, 14),legend_loc='upper left',fontsize=10):
    '''
    不同邻里尺度（邻元数），对应最优簇数的指数贡献度图表打印

    Parameters
    ----------
    featureScores_dict : dict
        聚类指数贡献度.
    save_fn : string
        保存路径名.

    Returns
    -------
    None.

    '''
    mpl.rcParams['font.sans-serif']=['DengXian'] #解决中文字符乱码问题
    
    kneighbors=list(featureScores_dict.keys())
    # print(kneighbors)
    featurescores_array=np.array([featureScores_dict[k]['Score'].to_list() for k in kneighbors])
    pValue_array=np.array([featureScores_dict[k]['p_value'].to_list() for k in kneighbors])
    factors=featureScores_dict[kneighbors[0]]['Factor'].to_list() 
    factors_mapping={'Green view index':'GVI',
                     'Sky view factor':'SVF',
                     'Ground view index':'GVP',
                     'Equilibrium degree':'ED',
                     'Perimeter area ratio(mn)':'PARA',
                     'Shape index(mn)':'SHAPE',
                     'Fractal dimension(mn)':'FRAC',
                     'Color richness index':'CRI',
                     'Key point size(0-10]':'KPSF (0-10]',
                     'Key point size(10-20]':'KPSF (10-20]',
                     'Key point size(30-40]':'KPSF(30-40]',
                     'Key point size(20-30]':'KPSF(20-30]'
                         }
    # print(factors)    
    y_min=featurescores_array.min()-100
    y_max=featurescores_array.max()+100    
    
    fig, ax=plt.subplots(1, 1, figsize=figsize)
    ax.set_prop_cycle(color=[
        '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a',
        '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
        '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d',
        '#17becf', '#9edae5']) 
    
    for i in ['top','right',]: #'bottom','left'
        ax.spines[i].set_visible(False)
    for i in ['bottom','left']: #'top','right',
        # ax.spines[i].set_visible(False)
        ax.spines[i].set_color('gray')
        
        
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()    
    fig.subplots_adjust(left=.06, right=.75, bottom=.02, top=.94)
    ax.set_xlim(min(kneighbors),max(kneighbors))
    ax.set_ylim(y_min,y_max) 
    # ax.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3) 
    # ax.tick_params(axis='both', which='both', labelsize=14,
    #                bottom=False, top=False, labelbottom=True,
    #                left=False, right=False, labelleft=True)    
    # factor_score_pValue={f:[s,p] for f,s,p in zip(factors, featurescores_array.T,pValue_array.T)}
    factor_score={f:s for f,s in zip(factors, featurescores_array.T)}
    # print(factor_score)
    # y_offsets={}
    y_offsets={'Key point size(30-40]':150,'Perimeter area ratio(mn)':150,'Equilibrium degree':100,'Fractal dimension(mn)':-100} 
    for f in factors:
        line,=ax.plot(kneighbors,factor_score[f] ,lw=2.5,label=factors_mapping[f]) 
        y_pos=factor_score[f][-1]
        # if f in y_offsets:
        #     y_pos+=y_offsets[f]
        # # ax.text(kneighbors[-1]+0.1,y_pos,f ,fontsize=14, color=line.get_color())
        # ax.text(kneighbors[-1]+0.1,y_pos,factors_mapping[f] ,fontsize=14, color=line.get_color())
    
    # ax.set_ylabel('Scores',fontsize=20)    
    # ax.set_xlabel('kneighbors',fontsize=20)
    ax.set_ylabel('分值',fontsize=fontsize)    
    ax.set_xlabel('邻元数',fontsize=fontsize)   
    
    ax.grid(False)    
    ax.tick_params('both', length=5, width=1, which='major',color='gray',direction='in')  
    # plt.ylim(-100,10000)
    # plt.xlim(0,160)
    plt.legend(loc=legend_loc,frameon=False)
    if save_fn:
        fig.savefig(save_fn,bbox_inches="tight",dpi=300)
    plt.show()  
    

def idxes_clustering_contribution_kneighbors_plot(featureScores_dict,figsize=(12, 14),fontsize=12,y_offsets={},x_offsets={},save_fn=None):
    '''
    不同邻里尺度（邻元数），对应最优簇数的指数贡献度图表打印

    Parameters
    ----------
    featureScores_dict : dict
        聚类指数贡献度.
    save_fn : string
        保存路径名.

    Returns
    -------
    None.

    '''
    import matplotlib.pyplot as plt    
    import numpy as np
    from pylab import mpl
    mpl.rcParams['font.sans-serif']=['DengXian'] #解决中文字符乱码问题
    
    kneighbors=list(featureScores_dict.keys())
    # print(kneighbors)
    featurescores_array=np.array([featureScores_dict[k]['Score'].to_list() for k in kneighbors])
    pValue_array=np.array([featureScores_dict[k]['p_value'].to_list() for k in kneighbors])
    factors=featureScores_dict[kneighbors[0]]['Factor'].to_list() 
    factors_mapping={'Green view index':'Seg_GVI',
                     'Sky view factor':'Seg_SVF',
                     'Ground view index':'Seg_GVI',
                     'Equilibrium degree':'Seg_ED',
                     'Perimeter area ratio(mn)':'Sky_PARA(mn)',
                     'Shape index(mn)':'Sky_SHAPE(mn)',
                     'Fractal dimension(mn)':'Sky_FRAC(mn)',
                     'Color richness index':'CRI',
                     'Key point size(0-10]':'KPSF (0-10]',
                     'Key point size(10-20]':'KPSF (10-20]',
                     'Key point size(30-40]':'KPSF(30-40]',
                     'Key point size(20-30]':'KPSF(20-30]'
                         }
    # print(factors)    
    y_min=featurescores_array.min()-100
    y_max=featurescores_array.max()+100    
    
    fig, ax=plt.subplots(1, 1, figsize=figsize)
    ax.set_prop_cycle(color=[
        '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a',
        '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
        '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d',
        '#17becf', '#9edae5']) 
    for i in ['top','right','bottom','left']:
        ax.spines[i].set_visible(False)
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()    
    fig.subplots_adjust(left=.06, right=.75, bottom=.02, top=.94)
    ax.set_xlim(min(kneighbors),max(kneighbors))
    ax.set_ylim(y_min,y_max) 
    ax.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3) 
    ax.tick_params(axis='both', which='both', labelsize=14,
                   bottom=False, top=False, labelbottom=True,
                   left=False, right=False, labelleft=True)    
    # factor_score_pValue={f:[s,p] for f,s,p in zip(factors, featurescores_array.T,pValue_array.T)}
    factor_score={f:s for f,s in zip(factors, featurescores_array.T)}
    # print(factor_score)
    # y_offsets={}
    # y_offsets={'Key point size(30-40]':150,'Perimeter area ratio(mn)':150,'Equilibrium degree':100,'Fractal dimension(mn)':-100} 
    for f in factors:
        line,=ax.plot(kneighbors,factor_score[f] ,lw=2.5) 
        y_pos=factor_score[f][-1]
        if f in y_offsets:
            y_pos+=y_offsets[f]
            
        x_pos=kneighbors[-1]+0.1
        if f in x_offsets:
            x_pos+=x_offsets[f]
        # ax.text(kneighbors[-1]+0.1,y_pos,f ,fontsize=14, color=line.get_color())
        ax.text(x_pos,y_pos,factors_mapping[f] ,fontsize=14, color=line.get_color())
    
    # ax.set_ylabel('Scores',fontsize=20)    
    # ax.set_xlabel('kneighbors',fontsize=20)
    ax.set_ylabel('分值',fontsize=fontsize)    
    ax.set_xlabel('邻元数',fontsize=fontsize)   
    if save_fn:
        fig.savefig(save_fn,bbox_inches="tight",dpi=300)
    plt.show()      
    
def gpd_plot(df,columns,save_path,**kwargs):
    '''
    不同列，GeoDataFrame地图打印

    Parameters
    ----------
    df : DataFrame
        待打印地图的数据.
    columns : list
        打印列名.
    save_path : string
        图表保存根目录.
    **kwargs : TYPE
        打印属性配置.

    Returns
    -------
    None.

    '''    
    plot_params={'figsize':(10,10),'markersize':2,'marker':'o','cmap':'rainbow'}
    plot_params.update(kwargs)
    for column in columns:
        df.plot(column=column,
                figsize=plot_params['figsize'],
                markersize=plot_params['markersize'],
                marker=plot_params['marker'],
                cmap=plot_params['cmap']) 
        plt.axis('off')
        plt.title(column)
        plt.savefig(os.path.join(save_path,'{}.png'.format(column)))
        plt.show()        
        plt.close() 
        
def idx_clustering(idxes_df,field,n_clusters=10,kneighbors_graph_n_neighbors=9):
    '''
    单个指数（字段）的聚类

    Parameters
    ----------
    idxes_df : DataFrame
        指数.
    field : string
        用于指数计算的列名.
    n_clusters : int, optional
        聚类数量. The default is 10.

    Returns
    -------
    idxes_df_gdf : GeoDataFrame
        单个指数（字段）的聚类.

    '''    
    pts_geometry=idxes_df[['geometry']]    
    pts_geometry[['x','y']]=pts_geometry.geometry.apply(lambda row:pd.Series([row.x,row.y]))
    pts_coordis=pts_geometry[['x','y']].to_numpy()
    
    # nbrs=NearestNeighbors(n_neighbors=kneighbors_graph_n_neighbors, algorithm='ball_tree').fit(pts_coordis)
    # connectivity=nbrs.kneighbors_graph(pts_coordis)
    connectivity=kneighbors_graph(pts_coordis,kneighbors_graph_n_neighbors,include_self=False)
    
    X=np.expand_dims(idxes_df[field].to_numpy(),axis=1)
    clustering=cluster.AgglomerativeClustering(connectivity=connectivity,n_clusters=n_clusters).fit(X)
    idxes_df['clustering_'+field]=clustering.labels_
    
    mean=idxes_df.groupby(['clustering_'+field])[field].mean() #.reset_index()
    idxes_df['clustering_'+field+'_mean']=idxes_df['clustering_'+field].map(mean.to_dict())
    
    wgs84=pyproj.CRS('EPSG:4326')
    idxes_df_gdf=gpd.GeoDataFrame(idxes_df,geometry=idxes_df.geometry,crs=wgs84)    
    
    clustering_=cluster.AgglomerativeClustering(connectivity=connectivity,) #n_clusters=n_clusters
    visualizer = KElbowVisualizer(clustering_, timings=False,size=(500, 500), ) #k=(4,12) metric='calinski_harabasz'
    visualizer.fit(X)    # Fit the data to the visualizer
    visualizer.show()    # Finalize and render the figure    
    
    return idxes_df_gdf
