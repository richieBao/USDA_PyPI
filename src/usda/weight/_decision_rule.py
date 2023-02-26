# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 20:32:32 2023

@author: richie bao 
"""
import pandas as pd
import numpy as np
import math
import itertools
from fractions import Fraction
from functools import reduce

def df_standardized_evaluation(df,columns=None):
    '''
    规范化每列或指定列的值，规范化公式为： v_{i j}=\frac{a_{i j}}{\sqrt{\sum_{i=1}^m a_{i j}^2}}

    Parameters
    ----------
    df : DataFrame
        数据.
    columns : list[str], optional
        待标准化的列，如未指定，则为所有列. The default is None.

    Returns
    -------
    std_eval_df : DaraFrame
        标注化后的数据.

    '''
    std_eval_df=df.copy()
    if columns:
        for col in columns:
            vals=df[col]
            std_eval_df[col]=vals/math.sqrt(np.power(vals,2).sum())
    else:
        for col in df.columns:
            vals=df[col]
            std_eval_df[col]=vals/math.sqrt(np.power(vals,2).sum())
            
    return std_eval_df


def PIS_NIS(df_std,pis):
    '''
    确定正理想解和负理想解，公式为：\begin{aligned} & V^{+}=\left\{v_1^{+}, v_2^{+}, \ldots, v_n^{+}\right\}=\left\{\left(\max _i v_{i j} \mid j \in J\right),\left(\min _i v_{i j} \mid j \in J^{\prime}\right), i=1,2, \ldots, m\right\} \\ & V^{-}=\left\{v_1^{-}, v_2^{-}, \ldots, v_n^{-}\right\}=\left\{\left(\min _i v_{i j} \mid j \in J\right),\left(\max _i v_{i j} \mid j \in J^{\prime}\right), i=1,2, \ldots, m\right\}\end{aligned}

    Parameters
    ----------
    df_std : DataFrame
        规范化后的数据.
    pis : dict[str:1/0]
        正理想解字典，1为越大越好；0为越小越好；负理想解与正理想解相反.

    Returns
    -------
    pis_nis_df : DataFrame
        正理想解和负理想解.

    '''
    pis_nis_dict={}
    compare_func=lambda array,pis: [max(array),min(array)] if pis else [min(array),max(array)] 
    for k,v in pis.items():
        pis_nis_dict[k]=compare_func(df_std[k].to_numpy(),v)

    pis_nis_df=pd.DataFrame(pis_nis_dict,index=['V+','V-'])
    
    return pis_nis_df

def closeness_pis_nis(df_std,pis,Wj,p=2):
    '''
    计算各方案到正负理想解的距离，公式为：\begin{aligned} & S_i^{+}=\left[\sum_{j=1}^n w_j^p \cdot\left(\left|v_{i j}-v_{i j}^{+}\right|\right)^p\right]^{\frac{1}{p}} \\ & S_i^{-}=\left[\sum_{j=1}^n w_j^p \cdot\left(\left|v_{i j}-v_{i j}^{-}\right|\right)^p\right]^{\frac{1}{p}}\end{aligned}
    各案的排序指标值（即综合评价指数），公式为： C_i^{+}=\frac{S_i^{-}}{S_i^{+}+S_i^{-}}

    Parameters
    ----------
    df_std : DataFrame
        规范化后的数据.
    pis : dict[str:1/0]
        正理想解字典，1为越大越好；0为越小越好；负理想解与正理想解相反.
    Wj : DataFrame/Series
        对应df_std数据的权重值.
    p : int, optional
        次方. The default is 2.

    Returns
    -------
    Si_pis_nis : DataFrame
        各方案到正负理想解的距离，及各案的排序指标值（即综合评价指数）结果.

    '''
    pis_nis=PIS_NIS(df_std,pis)

    df_std_array=df_std.to_numpy()
    V_pis=pis_nis.loc['V+'].to_numpy()
    W=Wj.to_numpy()
    Si_pis=(np.sum(np.power(W,p)*np.power(np.absolute(df_std_array- V_pis),p),axis=1))**(1/p)
    V_nis=pis_nis.loc['V-'].to_numpy()
    Si_nis=(np.sum(np.power(W,p)*np.power(np.absolute(df_std_array- V_nis),p),axis=1))**(1/p)
    
    Si_pis_nis=pd.DataFrame([Si_pis,Si_nis],index=['Si+','Si-']).T
    Si_pis_nis['Ci+']=Si_pis_nis.apply(lambda row:row['Si-']/(row['Si+']+row['Si-']),axis=1)
    Si_pis_nis['Rank']=Si_pis_nis['Ci+'].rank(method='max',ascending=False)

    return Si_pis_nis

class AHP:
    '''
    AHP（Analytic Hierarchy Process），层次分析法实现
    '''
    def __init__(self,criteria,alternatives):
        '''
        初始化值

        Parameters
        ----------
        criteria : list[str]
            决策准则列表.
        alternatives : list[str]
            备选方案列表.

        Returns
        -------
        None.

        '''
        self.criteria=criteria
        self.alternatives=alternatives
        
        self.criteria_pairs()      
        self.alternative_pairs()
        self.decision_matrix()

    def decision_matrix(self):
        '''
        构建决策矩阵

        Returns
        -------
        None.

        '''
        
        self.A=pd.DataFrame(np.nan,index=['Wj']+self.alternatives,columns=self.criteria) 
    
    def criteria_pairs(self):
        '''
        构建决策准则比较对列表，根据该返回值配置决策准则（准则层，level 2）对于目标层（level 1）的对比较值的顺序

        Returns
        -------
        None.

        '''
        
        self.criteria_pairs_lst=list(itertools.combinations(self.criteria,2))
        
    def alternative_pairs(self):
        '''
        构建备选方案比较对列表，根据该返回值配置各个决策准则准则层，level 2）对于方案层（level 3）的对比较值的顺序

        Returns
        -------
        None.

        '''
        self.alternative_pairs_lst=list(itertools.combinations(self.alternatives,2))        
        
    def pairwise_comparison_matrix(self,pairwise_comparison_vals,alternatives_criteria):
        '''
        构建比较对值矩阵，为正互反矩阵

        Parameters
        ----------
        pairwise_comparison_vals : list[str]
            比较对值列表，以字符串形式输入.
        alternatives_criteria : str
            为'alternatives'（'A'）准则层对目标层，或者'criteria' （'C'）准则层对方案层.

        Returns
        -------
        pairs_A_sorted : DataFrame
            返回比较对值矩阵，保持行列顺序与输入同.

        '''
        if alternatives_criteria=='alternatives' or alternatives_criteria=='A':
            cols=self.criteria
            pairs=self.criteria_pairs_lst
        elif alternatives_criteria=='criteria' or alternatives_criteria=='C':
            cols=self.alternatives   
            pairs=self.alternative_pairs_lst
        
        pairs_vals=[Fraction(i) for i in pairwise_comparison_vals]
        pairs_symmetrical_vals=[1/v for v in pairs_vals]       

        pairs_vals_dict={k:v for k,v in zip(pairs,pairs_vals)}        
        criteria_pairs_symmetrical_vals_dict={tuple(reversed(k)):v for k,v in zip(pairs,pairs_symmetrical_vals)}
        pairs_vals_dict.update(criteria_pairs_symmetrical_vals_dict)

        pairs_self_vals={(i,i):1 for i in cols}
        pairs_vals_dict.update(pairs_self_vals)        

        pairs_df=pd.Series(pairs_vals_dict)
        pairs_df.index=pd.MultiIndex.from_tuples(pairs_df.index)
        pairs_A=pairs_df.unstack(level=-1)
        pairs_A_sorted=pairs_A.reindex(index=cols,columns=cols)       
        
        return pairs_A_sorted
    
    def ahp_method(self,dataset,wd='m'):
        '''
        计算权重矩阵和一致性比例。算法包括算术平均法求权重和几何平均法求权重

        Parameters
        ----------
        dataset : 2darray/DataFrame
            两两比较矩阵（判断矩阵）.
        wd : str, optional
            算术平均法求权重为'mean'或'm';几何平均法求权重为'geometric'或'g'. The default is 'm'.

        Returns
        -------
        weights : list[float]
            权重值.
        cr : float
            一致性比例.

        '''
        inc_rat  = np.array([0, 0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49, 1.51, 1.48, 1.56, 1.57, 1.59])
        X        = np.copy(dataset)
        weights  = np.zeros(X.shape[1])
        
        if (wd == 'm' or wd == 'mean'):
            weights  = np.mean(X/np.sum(X, axis = 0), axis = 1) # weights=np.sum(X/np.sum(X,axis=0),axis=1)/X.shape[0]    
        elif (wd == 'g' or wd == 'geometric'):
            for i in range (0, X.shape[1]):
                weights[i] = reduce( (lambda x, y: x * y), X[i,:])**(1/X.shape[1])
            weights = weights/np.sum(weights)      
            
        vector   = np.sum(X*weights, axis = 1)/weights
        lamb_max = np.mean(vector)
        cons_ind = (lamb_max - X.shape[1])/(X.shape[1] - 1)
        cr       = cons_ind/inc_rat[X.shape[1]]
        
        return weights, cr   
    
    def weight(self,pairwise_comparison_matrix,wd='m'):
        '''
        调用ahp_method()方法计算权重矩阵。因为判断矩阵为Fraction形式，通过pd.eval转换为数值

        Parameters
        ----------
        pairwise_comparison_matrix : DataFrame
            两两比较矩阵（判断矩阵）.
        wd : str, optional
            算术平均法求权重为'mean'或'm';几何平均法求权重为'geometric'或'g'. The default is 'm'.

        Returns
        -------
        weights : list[float]
            权重值.
        rc : float
            一致性比例.

        '''
        weight,rc=self.ahp_method(pairwise_comparison_matrix.apply(pd.eval),wd=wd)
        
        return weight,rc
    
    def weight_matrix(self,criteria2goal_weight,criteria2alternatives_weight):
        '''
        根据两个层级的权重矩阵（lobal）计算全局（global）权重值

        Parameters
        ----------
        criteria2goal_weight : TYPE
            DESCRIPTION.
        criteria2alternatives_weight : TYPE
            DESCRIPTION.

        Returns
        -------
        weight_A : TYPE
            DESCRIPTION.

        '''
        local_w_stack=np.stack([v['weight'] for v in criteria2alternatives_weight.values()],axis=-1)
        weight_array=np.insert(local_w_stack,0,criteria2goal_weight,0)
        weight_A=pd.DataFrame(weight_array,index=['Wj']+self.alternatives,columns=self.criteria)
        global_priorities=np.sum(local_w_stack*criteria2goal_weight,axis=1)
        weight_A['global_priority']=np.insert(global_priorities,0,np.nan,0)
        
        return weight_A