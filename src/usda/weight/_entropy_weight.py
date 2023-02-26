# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 17:54:42 2023

@author: richie bao
"""
import pandas as pd
import numpy as np
import math


def df_Pij(df,columns=None):
    '''
    将DataFrame数据，每一列或指定列各个值除以每一列的和（概率）。通常用于信息熵的计算

    Parameters
    ----------
    df : DataFrame
        数据.
    columns : list[str], optional
        计算概率的列. The default is None.

    Returns
    -------
    Pij : DataFrame
        含计算结果的DataFrame格式数据.

    '''
    Pij=df.copy()
    if columns:
        for col in columns:
            Pij[col]=df[col]/sum(df[col])
    else:
        for col in df.columns:
            Pij[col]=df[col]/sum(df[col])
            
    return Pij

def df_entropy(vc,base=None):
    '''
    计算信息熵，公式为：E_j=\left\{\begin{array}{cc}-\frac{\sum_{i=1}^m p_{i j} \cdot \ln \left(p_{i j}\right)}{\ln (m)} & , p_{i j} \neq 0 \\ 0 & , p_{i j}=0\end{array}\right.
                      式中：p_{i j}=\frac{a_{i j}}{\sum_{i=1}^m a_{i j}}    
                         
    Parameters
    ----------
    vc : 1darray
        值列表或1维数组.
    base : int/e, optional
        基. The default is None.

    Returns
    -------
    Ej : float
        信息熵.

    '''
    base=math.e if base is None else base
    Ej_lst=[]
    for v in vc:
        if v!=0:
            Ej_lst.append(v*math.log(v))
        else:
            Ej_lst.append(0)

    Ej=-sum(Ej_lst)/math.log(len(vc))
    
    return Ej

def entropy_weight(df,columns=None,base=None):
    '''
    计算信息熵和基于信息熵的权重
    权重公式为：w_j=\frac{1-E_j}{\sum_{i=1}^m\left(1-E_j\right)}

    Parameters
    ----------
    df : DataFrame
        数据.
    columns : list[str], optional
        用于计算信息熵的列，如果为空，则计算全部列. The default is None.
    base : int/e, optional
        基. The default is None.

    Returns
    -------
    E_W : DataFrame
        每列信息熵和权重值.

    '''
    
    base=math.e if base is None else base
    Pij_df=df_Pij(df,columns=columns)

    entropy_lst=[]
    if columns:
        cols=columns
        for col in cols:
            entropy_lst.append(df_entropy(Pij_df[col].to_numpy(),base=base))        
    else:
        cols=df.columns
        for col in cols:
            entropy_lst.append(df_entropy(Pij_df[col].to_numpy(),base=base))

    E_W=pd.DataFrame(entropy_lst,columns=['Ej'],index=cols)
    E_W['1-Ej']=E_W['Ej'].apply(lambda x:1-x)
    E_W['Wj']=E_W['1-Ej'].apply(lambda x:x/sum(E_W['1-Ej']))
    
    return E_W

