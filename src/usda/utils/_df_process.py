# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 10:15:34 2023

@author: richie bao
"""
import pandas as pd
import numpy as np

def complete_dataframe_rowcols(dfs,val=0,col=True,row=True):
    '''
    完全互相补齐多个DataFrame列表的行列

    Parameters
    ----------
    dfs : list[DataFrame]
        多个DataFrame格式数据列表.
    val : any, optional
        补全行列填充值. The default is 0.
    col : bool, optional
        是否补齐列. The default is True.
    row : bool, optional
        是否补齐行. The default is True.

    Returns
    -------
    dfs_row_completed : list[DataFrame]
        补齐行列的DataFrame列表.

    '''
    flatten_lst=lambda lst: [m for n_lst in lst for m in flatten_lst(n_lst)] if type(lst) is list else [lst]

    if col:
        columns=set(flatten_lst([df.columns.to_list() for df in dfs]))   
        dfs_col_completed=[]
        for df in dfs:     
            df[list(columns-set(df.columns.to_list()))]=val
            df.sort_index(axis=1, inplace=True)
            dfs_col_completed.append(df)
    else:
        dfs_col_completed=dfs        
    
    if row:
        rows=set(flatten_lst([df.index.to_list() for df in dfs_col_completed])) 
        dfs_row_completed=[]
        for df in dfs_col_completed:
            df2=pd.DataFrame([[val]*df.shape[1]],columns=df.columns,index=list(rows-set(df.index.to_list())))
            df=pd.concat([df,df2])
            df.sort_index(axis=0, inplace=True)
            dfs_row_completed.append(df)
        
    return dfs_row_completed

def xy_to_matrix(xy):
    '''
    将成对（类，对象）对应的值转换为矩阵形式；与matrix_to_xy()互逆
    by @jezrael

    Parameters
    ----------
    xy : DataFrame
        第1列为行索引类，第2列为列索引类.

    Returns
    -------
    df : DataFrame
        以类对象为行和列，并一一对应，为矩阵形式.

    '''
    index,columns,values=xy.columns
    df=xy.pivot(index=index,columns=columns,values=values).fillna(0)
    df_vals=df.to_numpy()
    df=pd.DataFrame(
        np.triu(df_vals, 1) + df_vals.T, index=df.index, columns=df.index
    )
    return df

def matrix_to_xy(df, columns=None, reset_index=False):
    '''
    将矩阵形式DataFrame转换为成对（类，对象）层级索引（hierarchical index）对应值形式；与xy_to_matrix()互逆
    by @jezrael
    
    Parameters
    ----------
    df : DataFrame
        成对（类，对象）为列和行的矩阵.
    columns : list, optional
        配置列名列表. The default is None.
    reset_index : bool, optional
        Reset the index, or a level of it. The default is False.

    Returns
    -------
    xy : DataFrame
        DESCRIPTION.

    '''
    bool_index=np.triu(np.ones(df.shape)).astype(bool)
    xy=(
        df.where(bool_index).stack().reset_index()
        if reset_index
        else df.where(bool_index).stack()
    )
    if reset_index:
        xy.columns=columns or ["row", "col", "val"]
    return xy