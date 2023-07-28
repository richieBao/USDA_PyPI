# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 16:50:13 2023

@author: richie bao
"""
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 10:28:35 2021
Updated on Tue Feb  1 19:50:14 2022

@author: Richie Bao-caDesign设计(cadesign.cn)
"""
from tqdm import tqdm
import pickle,math
import pandas as pd
import numpy as np
import geopandas as gpd

from sklearn.neighbors import NearestNeighbors
from sklearn import cluster  
from yellowbrick.cluster import KElbowVisualizer    
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.neighbors import kneighbors_graph

from matplotlib import cm


poi_classificationName_={
        "美食":"delicacy",
        "酒店":"hotel",
        "购物":"shopping",
        "生活服务":"lifeService",
        "丽人":"beauty",
        "旅游景点":"spot",
        "休闲娱乐":"entertainment",
        "运动健身":"sports",
        "教育培训":"education",
        "文化传媒":"media",
        "医疗":"medicalTreatment",
        "汽车服务":"carService",
        "交通设施":"trafficFacilities",
        "金融":"finance",
        "房地产":"realEstate",
        "公司企业":"corporation",
        "政府机构":"government",
        "出入口":"entrance",
        "自然地物":"naturalFeatures",        
        "行政地标":"administrativeLandmarks",
        "门址":"address",       
        "道路":"road",
        "公交车站":"busStop",
        "地铁站":"subwayStop",
        "商圈":"businessDistrict",
        }

poi_classificationName={
        0:"delicacy",
        1:"hotel",
        2:"shopping",
        3:"lifeService",
        4:"beauty",
        5:"spot",
        6:"entertainment",
        7:"sports",
        8:"education",
        9:"media",
        10:"medicalTreatment",
        11:"carService",
        12:"trafficFacilities",
        13:"finance",
        14:"realEstate",
        15:"corporation",
        16:"government",
        17:"entrance",
        18:"naturalFeatures", 
        19:"administrativeLandmarks",
        20:"address",
        21:"road",
        22:"busStop",
        23:"subwayStop",
        24:"businessDistrict"        
        }
poi_classificationName_reverse={v:k for k,v in poi_classificationName.items()}

def street_poi_structure(poi,position,distance=350,save_fn=None):
    '''
    计算给定道路上点，指定半径内，POI的组成结构，包括：数量、一级分类占比、频数、信息熵

    Parameters
    ----------
    poi : GeoDataFrame
        poi数据，含提取的一级分类标签，列名为'level_0'.
    position : GeoDataFrame
        道路采样点，用排序的'geometry'列.
    save_fn : string
        数据保存路径名 为.pkl.
    distance : numerical val, optional
        缓冲半径. The default is 350.

    Returns
    -------
    pos_poi_idxes_gdf : GeoDataFrame
        含数量、一级分类占比、信息熵.
    pos_poi_feature_vector_gdf : GeoDataFrame
        频数信息.

    '''
    poi_num=len(poi_classificationName.keys())    
    feature_vector=np.zeros(poi_num)
    
    poi_=poi.copy(deep=True)
    pos_poi_dict={}
    # pos_poi_idxes_df=pd.DataFrame(columns=['geometry','frank_e','num'])
    pos_poi_idxes_lst=[]
    # pos_poi_feature_vector_df=pd.DataFrame(columns=['geometry']+list(range(poi_num)))
    pos_poi_feature_vector_lst=[]
    for idx,row in tqdm(position.iterrows(),total=position.shape[0]):
        poi_['within']=poi_.geometry.apply(lambda pt: pt.within(row.geometry.buffer(distance)))
        poi_selection_df=poi_[poi_['within']==True]
        counts=poi_selection_df.level_0.value_counts().to_dict()
        num=len(poi_selection_df)
        counts_percent={k:v/num for k,v in counts.items()}        
        ve=0.0
        for v in counts_percent.values():
            if v!=0.:
                ve-=v*math.log(v)
        max_entropy=math.log(num)
        frank_e=ve/max_entropy*100        
        
        for k,v in counts.items(): #计算特征聚类出现的频数/直方图
            poi_name=k.split("_")[-1]
            poi_idx=poi_classificationName_reverse[poi_name]
            feature_vector[poi_idx]=v        
        pos_poi_dict.update({idx:{'fn_stem':row.fn_stem, 'fn_key':row.fn_key, 'fn_idx':row.fn_idx ,'counts':counts,'counts_percent':counts_percent,'feature_vector':feature_vector,'num':num,'frank_e':frank_e,'geometry':row.geometry}})
        # pos_poi_idxes_df=pos_poi_idxes_df.append({'fn_stem':row.fn_stem, 'fn_key':row.fn_key, 'fn_idx':row.fn_idx,'geometry':row.geometry,'frank_e':frank_e,'num':num},ignore_index=True)
        pos_poi_idxes_lst.append({'fn_stem':row.fn_stem, 'fn_key':row.fn_key, 'fn_idx':row.fn_idx,'geometry':row.geometry,'frank_e':frank_e,'num':num})
        
        feature_vector_dict={i:feature_vector[i] for i in range(len(feature_vector))}
        feature_vector_dict.update({'geometry':row.geometry,'fn_stem':row.fn_stem, 'fn_key':row.fn_key, 'fn_idx':row.fn_idx,})
        # pos_poi_feature_vector_df=pos_poi_feature_vector_df.append(feature_vector_dict,ignore_index=True)
        pos_poi_feature_vector_lst.append(feature_vector_dict)
        
        # if idx==3:break      
    pos_poi_idxes_df=pd.DataFrame(pos_poi_idxes_lst,columns=['geometry','frank_e','num'])
    pos_poi_feature_vector_df=pd.DataFrame(pos_poi_feature_vector_lst,columns=['geometry']+list(range(poi_num))) 
    pos_poi_idxes_gdf=gpd.GeoDataFrame(pos_poi_idxes_df,geometry=pos_poi_idxes_df.geometry,crs=position.crs)   
    pos_poi_idxes_gdf['num_diff']=pos_poi_idxes_gdf.num.diff()
    pos_poi_feature_vector_gdf=gpd.GeoDataFrame(pos_poi_feature_vector_df,geometry=pos_poi_feature_vector_df.geometry,crs=position.crs) 
    
    if save_fn:
        with open(save_fn,'wb') as f:
            pickle.dump(pos_poi_dict,f)    
        
    return pos_poi_idxes_gdf,pos_poi_feature_vector_gdf

def poi_feature_clustering(feature_vector,fields,save_fn,n_clusters=7,n_neighbors=10,feature_analysis=True):
    '''
    poi聚类及最优簇数和特征贡献度计算

    Parameters
    ----------
    feature_vector : GeoDataFrame
        频数信息.
    fields : list
        指定参与聚类计算的列名.
    save_fn : string
        特征贡献度保存路径名.
    n_clusters : int, optional
        聚类数量. The default is 7.
    n_neighbors:int
        邻元数
    feature_analysis : bool, optional
        是否计算特征贡献度. The default is True.

    Returns
    -------
    feature_vector : GeoDataFrame
        返回聚类.

    '''    
    pts_geometry=feature_vector[['geometry']]    
    pts_geometry[['x','y']]=pts_geometry.geometry.apply(lambda row:pd.Series([row.x,row.y]))
    pts_coordis=pts_geometry[['x','y']].to_numpy()
    connectivity=kneighbors_graph(pts_coordis,n_neighbors,include_self=False)
    
    X_=feature_vector[fields].to_numpy()
    X=normalize(X_,axis=0, norm='max')    

    clustering=cluster.AgglomerativeClustering(connectivity=connectivity,n_clusters=n_clusters).fit(X)
    feature_vector['clustering']=clustering.labels_
    
    #_________________________________________________________________________
    if feature_analysis==True:
        y=clustering.labels_
        selector=SelectKBest(score_func=f_classif, k=len(fields)) #score_func=chi2    
        selector.fit(X,y)
        
        dfscores = pd.DataFrame(selector.scores_)
        dfpvalues=pd.DataFrame(selector.pvalues_)
        dfcolumns = pd.DataFrame(fields)  
        featureScores = pd.concat([dfcolumns,dfscores,dfpvalues],axis=1)
        featureScores.columns = ['Factor','Score','p_value']  #naming the dataframe columns
        featureScores['Factor']=featureScores['Factor'].apply(lambda row:int(row))
        featureScores['poi_name']=featureScores['Factor'].map(poi_classificationName)
        featureScores=featureScores.sort_values(by=['Score']).round(3)
        print(featureScores)
        featureScores.to_excel(save_fn) 
        
        featureScores_=featureScores.set_index('Factor')    
        featureScores_.nlargest(len(fields),'Score').Score.plot(kind='barh',figsize=(30,20),fontsize=38)
        featureScores_.Score.plot(kind='barh')
        plt.show()    
        
        clustering_=cluster.AgglomerativeClustering(connectivity=connectivity,) #n_clusters=n_clusters
        visualizer = KElbowVisualizer(clustering_, timings=False,size=(500, 500), k=(4,12)) #k=(4,12) metric='calinski_harabasz'
        visualizer.fit(X)    # Fit the data to the visualizer
        # visualizer.show(outpath="./graph/tl_poi_clustering_KEIbow_.png")    # Finalize and render the figure   
    return feature_vector 

def clustering_POI_stats(df,save_fn):
    '''
    poi聚类业态频数打印

    Parameters
    ----------
    df : DataFrame
        聚类业态类频数.
    save_fn : string
        图像保存路径.

    Returns
    -------
    None.

    '''    
    stats_sum=df.groupby(['clustering_POI']).sum()
    stats_sum=stats_sum.rename(columns={str(k):v for k,v in poi_classificationName.items()})
    print(stats_sum)   
    fig, ax=plt.subplots(figsize=(30, 12),) #figsize=(40, 20),
    cmap=cm.get_cmap('tab20') # Colour map (there are many others)
    plot=stats_sum.plot(kind='bar',stacked=True, legend=True,ax=ax,rot=0,fontsize=35,cmap=cmap)
    ax.set_facecolor("w")
    plt.legend(prop={"size":20},bbox_to_anchor=(1, 1))
    plt.savefig(save_fn,dpi=300,bbox_inches="tight")
    plt.show()

# if __name__=="__main__":
#     import geopandas as gpd
#     import pickle
#     import pandas as pd
#     import sys,os
#     sys.path.append('..')      
#     from database import postSQL2gpd,gpd2postSQL,cfg_load_yaml 
#     parent_path=os.path.dirname(os.getcwd())
#     cfg=cfg_load_yaml('../config.yml')  
    
#     UN=cfg["postgreSQL"]["myusername"]
#     PW=cfg["postgreSQL"]["mypassword"]
#     DB=cfg["postgreSQL"]["mydatabase"]  
#     GC='geometry'  
    
#     #A.计算街道行业分类服务空间组成结构，包括数量、一级分类占比、频数、信息熵等
#     poi_gdf=postSQL2gpd(table_name='poi',geom_col=GC,myusername=UN,mypassword=PW,mydatabase=DB)
#     poi_gdf=poi_gdf.to_crs(cfg['xian_epsg'])
#     poi_gdf=poi_gdf[pd.notnull(poi_gdf['detail_info_tag'])]
#     poi_gdf['level_0']=poi_gdf.detail_info_tag.apply(lambda row:poi_classificationName_[row.split(";")[0]])
    
#     tourLine_panorama_object_percent_table_name=cfg['vanishing_position']['tourLine_panorama_object_percent_table_name']  
#     tourLine_panorama_object_percent_gdf=postSQL2gpd(table_name=tourLine_panorama_object_percent_table_name,geom_col=GC,myusername=UN,mypassword=PW,mydatabase=DB)
#     coordi_df=tourLine_panorama_object_percent_gdf.sort_values(by='fn_idx')
#     coordi_df=coordi_df.to_crs(cfg['xian_epsg'])    
    
#     pos_poi_dict_pkl_fn=os.path.join(parent_path,cfg['POI_street_feature']['pos_poi_dict_pkl_fn'])
#     buffer_radius=cfg['POI_street_feature']['buffer_radius']
#     pos_poi_idxes_gdf,pos_poi_feature_vector_gdf=street_poi_structure(poi=poi_gdf,position=coordi_df,save_fn=pos_poi_dict_pkl_fn,distance=buffer_radius)
#     gpd2postSQL(pos_poi_idxes_gdf,table_name='pos_poi_idxes',myusername=UN,mypassword=PW,mydatabase=DB) 
#     gpd2postSQL(pos_poi_feature_vector_gdf,table_name='pos_poi_feature_vector',myusername=UN,mypassword=PW,mydatabase=DB) 
    
#     #B.poi聚类及最优簇数和特征贡献度计算
#     pos_poi_feature_vector_gdf=postSQL2gpd(table_name='pos_poi_feature_vector',geom_col=GC,myusername=UN,mypassword=PW,mydatabase=DB)
#     fields=[str(i) for i in poi_classificationName.keys()]
#     featureScores_save_fn=os.path.join(parent_path,cfg['POI_street_feature']['featureScores_save_fn'])
#     n_clusters=12 #7,12
#     n_neighbors=10
#     feature_vector=poi_feature_clustering(pos_poi_feature_vector_gdf,fields,featureScores_save_fn,n_clusters=n_clusters,n_neighbors=n_neighbors,feature_analysis=True)
#     gpd2postSQL(feature_vector,table_name='pos_poi_feature_vector_{}'.format(n_clusters),myusername=UN,mypassword=PW,mydatabase=DB) 
    
#     #C.poi聚类业态频数打印
#     feature_vector=postSQL2gpd(table_name='pos_poi_feature_vector_{}'.format(n_clusters),geom_col=GC,myusername=UN,mypassword=PW,mydatabase=DB)
#     pos_poi_feature_vector_gdf['clustering_POI']=feature_vector['clustering']
#     clustering_POI_stats_save_fn=os.path.join(parent_path,cfg['POI_street_feature']['clustering_POI_stats_save_fn'])
#     clustering_POI_stats(pos_poi_feature_vector_gdf[fields+['clustering_POI']],clustering_POI_stats_save_fn)
