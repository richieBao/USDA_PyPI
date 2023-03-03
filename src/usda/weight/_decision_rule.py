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
import matplotlib.pyplot as plt
from collections import defaultdict

from ..meta_heuristics import grey_wolf_optimizer,genetic_algorithm

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
        迁移于： pyDecision库
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

class F_AHP:
    '''
    F-AHP （Fuzzy analytic hierarchy process） ，模糊层次分析法
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
        
    def pairwise_comparison_TFN_matrix(self,pairwise_comparison_vals,alternatives_criteria):
        '''
        构建比较对值矩阵，为TFN（triangular fuzzy number）正互反矩阵

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
            cols=self.alternatives   
            pairs=self.alternative_pairs_lst
        elif alternatives_criteria=='criteria' or alternatives_criteria=='C':
            cols=self.criteria
            pairs=self.criteria_pairs_lst     

        pairs_vals=[tuple([Fraction(j) for j in i]) for i in pairwise_comparison_vals]        
        pairs_symmetrical_vals=[[1/v_ for v_ in v] for v in pairs_vals]   
        pairs_symmetrical_vals=[tuple([v[-1],v[1],v[0]]) for v in pairs_symmetrical_vals]
        
        pairs_vals_dict={k:v for k,v in zip(pairs,pairs_vals)}  
        criteria_pairs_symmetrical_vals_dict={tuple(reversed(k)):v for k,v in zip(pairs,pairs_symmetrical_vals)}
        pairs_vals_dict.update(criteria_pairs_symmetrical_vals_dict)
        
        pairs_self_vals={(i,i):(1,1,1) for i in cols}
        pairs_vals_dict.update(pairs_self_vals)        

        pairs_df=pd.Series(pairs_vals_dict)
        pairs_df.index=pd.MultiIndex.from_tuples(pairs_df.index)
        pairs_A=pairs_df.unstack(level=-1)
        pairs_A_sorted=pairs_A.reindex(index=cols,columns=cols)       
        
        return pairs_A_sorted
            
    def fuzzy_ahp_method(self,dataset):
        '''
        计算权重矩阵和一致性比例。算法为几何平均法求权重
        迁移于： pyDecision库
        Parameters
        ----------
        dataset : 2darray/DataFrame
            两两比较矩阵（判断矩阵）.

        Returns
        -------
        f_w : list[float]
            fuzzy_weights.
        d_w : list[float]
            defuzzified_weights.
        n_w : list[float]
           normalized_weights.
        cr : float
            CR一致性比率.

        '''
        row_sum = []
        s_row   = []
        f_w     = []
        d_w     = []
        inc_rat  = np.array([0, 0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49, 1.51, 1.48, 1.56, 1.57, 1.59])
        X       = [(item[0] + 4*item[1] + item[2])/6 for i in range(0, len(dataset)) for item in dataset[i] ]
        X       = np.asarray(X)
        X       = np.reshape(X, (len(dataset), len(dataset)))
        for i in range(0, len(dataset)):
            a, b, c = 1, 1, 1
            for j in range(0, len(dataset[i])):
                d, e, f = dataset[i][j]
                a, b, c = a*d, b*e, c*f
            row_sum.append( (a, b, c) )
        L, M, U = 0, 0, 0
        for i in range(0, len(row_sum)):
            a, b, c = row_sum[i]
            a, b, c = a**(1/len(dataset)), b**(1/len(dataset)), c**(1/len(dataset))
            s_row.append( ( a, b, c ) )
            L = L + a
            M = M + b
            U = U + c
        for i in range(0, len(s_row)):
            a, b, c = s_row[i]
            a, b, c = a*(U**-1), b*(M**-1), c*(L**-1)
            f_w.append( ( a, b, c ) )
            d_w.append( (a + b + c)/3 )
        n_w      = [item/sum(d_w) for item in d_w]
        vector   = np.sum(X*n_w, axis = 1)/n_w
        lamb_max = np.mean(vector)
        cons_ind = (lamb_max - X.shape[1])/(X.shape[1] - 1)
        cr       = cons_ind/inc_rat[X.shape[1]]
        return f_w, d_w, n_w, cr
    
    def weight(self,pairwise_comparison_matrix):
        '''
        调用fuzzy_ahp_method方法计算权重矩阵

        Parameters
        ----------
        pairwise_comparison_matrix : DataFrame
            两两比较矩阵（判断矩阵）.

        Returns
        -------
        f_w : list[float]
            fuzzy_weights.
        d_w : list[float]
            defuzzified_weights.
        n_w : list[float]
           normalized_weights.
        cr : float
            CR一致性比率.

        '''

        pcm=pairwise_comparison_matrix.to_numpy()
        f_w, d_w, n_w, cr=self.fuzzy_ahp_method(pcm)
        
        return f_w, d_w, n_w, cr
    
    def weight_matrix(self,criteria2goal_weight,local_w_stack):
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
        weight_array=np.insert(local_w_stack,0,criteria2goal_weight,0)
        weight_A=pd.DataFrame(weight_array,index=['Wj']+self.alternatives,columns=self.criteria)
        global_priorities=np.sum(local_w_stack*criteria2goal_weight,axis=1)
        weight_A['global_priority']=np.insert(global_priorities,0,np.nan,0)
        
        return weight_A        
    
# Function: Rank 
def ranking(flow):    
    '''
    MCDM的	ARAS （Additive Ratio Assessment ）算法结果图表

    Parameters
    ----------
    flow : array
        为aras_method函数返回结果.

    Returns
    -------
    None.

    '''
    rank_xy = np.zeros((flow.shape[0], 2))
    for i in range(0, rank_xy.shape[0]):
        rank_xy[i, 0] = 0
        rank_xy[i, 1] = flow.shape[0]-i           
    for i in range(0, rank_xy.shape[0]):
        plt.text(rank_xy[i, 0],  rank_xy[i, 1], 'a' + str(int(flow[i,0])), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0.8, 1.0, 0.8),))
    for i in range(0, rank_xy.shape[0]-1):
        plt.arrow(rank_xy[i, 0], rank_xy[i, 1], rank_xy[i+1, 0] - rank_xy[i, 0], rank_xy[i+1, 1] - rank_xy[i, 1], head_width = 0.01, head_length = 0.2, overhang = 0.0, color = 'black', linewidth = 0.9, length_includes_head = True)
    axes = plt.gca()
    axes.set_xlim([-1, +1])
    ymin = np.amin(rank_xy[:,1])
    ymax = np.amax(rank_xy[:,1])
    if (ymin < ymax):
        axes.set_ylim([ymin, ymax])
    else:
        axes.set_ylim([ymin-1, ymax+1])
    plt.axis('off')
    plt.show() 
    return

# Function: ARAS (Additive Ratio Assessment)
def aras_method(dataset, weights, criterion_type, graph = True):
    '''
    MCDM的	ARAS （Additive Ratio Assessment ）算法
    参考：Zavadskas, E. K. & Turskis, Z. A new additive ratio assessment (ARAS) method in multicriteria decision-making. Technological and Economic Development of Economy 16, 159–172 (2010).
    迁移于pyDecision库

    Parameters
    ----------
    dataset : array/DataFrame
        决策矩阵.
    weights : list[float]
        决策准则权重.
    criterion_type : list[str]
        优化方向.
    graph : bool, optional
        是否打印图表. The default is True.

    Returns
    -------
    flow : array[int,float]
        备选方案得分并降序排序.

    '''
    X   = np.copy(dataset)/1.0
    bst = np.zeros(X.shape[1])
    bsm = np.zeros(X.shape[1])
    for j in range(0, X.shape[1]):
        if ( criterion_type[j] == 'max'):
            bst[j] = np.max(X[:, j])
            bsm[j] = bst[j] + np.sum(X[:, j])
        elif ( criterion_type[j] == 'min'):
            bst[j]  = 1/np.min(X[:, j])
            X[:, j] = 1/X[:, j]
            bsm[j]  = bst[j] + np.sum(X[:, j])
    for j in range(0, X.shape[1]):
        bst[j] = bst[j]/ bsm[j]
        for i in range(0, X.shape[0]):
            X[i, j] = X[i, j]/ bsm[j]
    X    = X*weights
    bst  = bst*weights
    n_0  = np.sum(bst)
    n_i  = np.sum(X, axis = 1)
    k_i  = n_i/n_0
    flow = np.copy(k_i)
    flow = np.reshape(flow, (k_i.shape[0], 1))
    flow = np.insert(flow, 0, list(range(1, k_i.shape[0]+1)), axis = 1)
    for i in range(0, flow.shape[0]):
        print('a' + str(int(flow[i,0])) + ': ' + str(round(flow[i,1], 3))) 
    if (graph == True):
        flow = flow[np.argsort(flow[:, 1])]
        flow = flow[::-1]
        ranking(flow)
    return flow

def bw_method(mic, lic, size = 50, iterations = 150,verbose=1):
    '''
    引自：pyDecision库：https://github.com/Valdecy/pyDecision

    Parameters
    ----------
    mic : TYPE
        DESCRIPTION.
    lic : TYPE
        DESCRIPTION.
    size : TYPE, optional
        DESCRIPTION. The default is 50.
    iterations : TYPE, optional
        DESCRIPTION. The default is 150.
    verbose : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    best      = np.where(mic == 1)[0][0]
    worst     = np.where(lic == 1)[0][0]
    pairs_b   = [(best, i)  for i in range(0, mic.shape[0])]
    pairs_w   = [(i, worst) for i in range(0, mic.shape[0]) if (i, worst) not in pairs_b]
    def target_function(variables):
        eps       = [float('+inf')]
        for pair in pairs_b:
            i, j = pair
            diff = abs(variables[i] - variables[j]*mic[j])
            if ( i != j):
                eps.append(diff)
        for pair in pairs_w:
            i, j = pair
            diff = abs(variables[i] - variables[j]*lic[j])
            if ( i != j):
                eps.append(diff)
        if ( np.sum(variables) == 0):
            eps = float('+inf')
        else:
            eps = max(eps[1:])
        return eps
    weights = grey_wolf_optimizer(pack_size = size, min_values = [0.01]*len(mic), max_values = [1]*len(mic), iterations = iterations, target_function = target_function,verbose=verbose)
    weights = weights[0][:-1]/sum(weights[0][:-1])

    return weights

def dematel_method(dataset, size_x = 10, size_y = 10):  
    '''
    引自：pyDecision库：https://github.com/Valdecy/pyDecision

    Parameters
    ----------
    dataset : TYPE
        DESCRIPTION.
    size_x : TYPE, optional
        DESCRIPTION. The default is 10.
    size_y : TYPE, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    D_plus_R : TYPE
        DESCRIPTION.
    D_minus_R : TYPE
        DESCRIPTION.
    weights : TYPE
        DESCRIPTION.

    '''
    row_sum = np.sum(dataset, axis = 1)
    max_sum = np.max(row_sum)
    X = dataset/max_sum
    Y = np.linalg.inv(np.identity(dataset.shape[0]) - X) 
    T = np.matmul (X, Y)
    D = np.sum(T, axis = 1)
    R = np.sum(T, axis = 0)
    D_plus_R   = D + R # Most Importante Criteria
    D_minus_R  = D - R # +Influencer Criteria, - Influenced Criteria
    weights    = D_plus_R/np.sum(D_plus_R)
    print('QUADRANT I has the Most Important Criteria (Prominence: High, Relation: High)') 
    print('QUADRANT II has Important Criteira that can be Improved by Other Criteria (Prominence: Low, Relation: High)') 
    print('QUADRANT III has Criteria that are not Important (Prominence: Low, Relation: Low)')
    print('QUADRANT IV has Important Criteria that cannot be Improved by Other Criteria (Prominence: High, Relation: Low)')
    print('')
    plt.figure(figsize = [size_x, size_y])
    plt.style.use('ggplot')
    for i in range(0, dataset.shape[0]):
        if (D_minus_R[i] >= 0 and D_plus_R[i] >= np.mean(D_plus_R)):
            plt.text(D_plus_R[i],  D_minus_R[i], 'g'+str(i+1), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0.7, 1.0, 0.7),)) 
            print('g'+str(i+1)+': Quadrant I')
        elif (D_minus_R[i] >= 0 and D_plus_R[i] < np.mean(D_plus_R)):
            plt.text(D_plus_R[i],  D_minus_R[i], 'g'+str(i+1), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (1.0, 1.0, 0.7),))
            print('g'+str(i+1)+': Quadrant II')
        elif (D_minus_R[i] < 0 and D_plus_R[i] < np.mean(D_plus_R)):
            plt.text(D_plus_R[i],  D_minus_R[i], 'g'+str(i+1), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (1.0, 0.7, 0.7),)) 
            print('g'+str(i+1)+': Quadrant III')
        else:
            plt.text(D_plus_R[i],  D_minus_R[i], 'g'+str(i+1), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0.7, 0.7, 1.0),)) 
            print('g'+str(i+1)+': Quadrant IV')
    axes = plt.gca()
    xmin = np.amin(D_plus_R)
    if (xmin > 0):
        xmin = 0
    xmax = np.amax(D_plus_R)
    if (xmax < 0):
        xmax = 0
    axes.set_xlim([xmin-1, xmax+1])
    ymin = np.amin(D_minus_R)
    if (ymin > 0):
        ymin = 0
    ymax = np.amax(D_minus_R)
    if (ymax < 0):
        ymax = 0
    axes.set_ylim([ymin-1, ymax+1]) 
    plt.axvline(x = np.mean(D_plus_R), linewidth = 0.9, color = 'r', linestyle = 'dotted')
    plt.axhline(y = 0, linewidth = 0.9, color = 'r', linestyle = 'dotted')
    plt.xlabel('Prominence (D + R)')
    plt.ylabel('Relation (D - R)')
    plt.show()
    return D_plus_R, D_minus_R, weights

# Function: IDOCRIW
def idocriw_method(dataset, criterion_type, size = 20, gen = 12000, graph = True,verbose = 1):
    '''
    引自：pyDecision库：https://github.com/Valdecy/pyDecision

    Parameters
    ----------
    dataset : TYPE
        DESCRIPTION.
    criterion_type : TYPE
        DESCRIPTION.
    size : TYPE, optional
        DESCRIPTION. The default is 20.
    gen : TYPE, optional
        DESCRIPTION. The default is 12000.
    graph : TYPE, optional
        DESCRIPTION. The default is True.
    verbose : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    X    = np.copy(dataset)
    X    = X/X.sum(axis = 0)
    X_ln = np.copy(dataset)
    X_r  = np.copy(dataset)
    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[1]):
            X_ln[i,j] = X[i,j]*math.log(X[i,j])
    d    = np.zeros((1, X.shape[1]))
    w    = np.zeros((1, X.shape[1]))
    for i in range(0, d.shape[1]):
        d[0,i] = 1-( -1/(math.log(d.shape[1]))*sum(X_ln[:,i])) 
    for i in range(0, w.shape[1]):
        w[0,i] = d[0,i]/d.sum(axis = 1)
    for i in range(0, len(criterion_type)):
        if (criterion_type[i] == 'min'):
           X_r[:,i] = dataset[:,i].min() / X_r[:,i]
    X_r   = X_r/X_r.sum(axis = 0)
    #a_min = X_r.min(axis = 0)       
    a_max = X_r.max(axis = 0) 
    A     = np.zeros(dataset.shape)
    np.fill_diagonal(A, a_max)
    for k in range(0, A.shape[0]):
        i, _ = np.where(X_r == a_max[k])
        i    = i[0]
        for j in range(0, A.shape[1]):
            A[k, j] = X_r[i, j]
    #a_min_ = A.min(axis = 0)       
    a_max_ = A.max(axis = 0) 
    P      = np.copy(A)    
    for i in range(0, P.shape[1]):
        P[:,i] = (-P[:,i] + a_max_[i])/a_max[i]
    WP     = np.copy(P)
    np.fill_diagonal(WP, -P.sum(axis = 0))
    
    ################################################
    def target_function(variable = [0]*WP.shape[1]):
        variable = [variable[i]/sum(variable) for i in range(0, len(variable))]
        WP_s     = np.copy(WP)
        for i in range(0, WP.shape[0]):
            for j in range(0, WP.shape[1]):
                WP_s[i, j] = WP_s[i, j]*variable[j]
        total = abs(WP_s.sum(axis = 1)) 
        total = sum(total) 
        return total
    ################################################
    
    solution = genetic_algorithm(population_size = size, mutation_rate = 0.1, elite = 1, min_values = [0]*WP.shape[1], max_values = [1]*WP.shape[1], eta = 1, mu = 1, generations = gen, target_function = target_function,verbose = verbose)
    solution = solution[:-1]
    solution = solution/sum(solution)
    print(f'solution:{solution}')
    w_       = np.copy(w)
    w_       = w_*solution
    w_       = w_/w_.sum()
    w_       = w_.T
    for i in range(0, w_.shape[0]):
        print('a' + str(i+1) + ': ' + str(round(w_[i][0], 4)))
    if ( graph == True):
        flow = np.copy(w_)
        flow = np.reshape(flow, (w_.shape[0], 1))
        flow = np.insert(flow, 0, list(range(1, w_.shape[0]+1)), axis = 1)
        flow = flow[np.argsort(flow[:, 1])]
        flow = flow[::-1]
        ranking(flow)
    return w_
    
###############################################################################
# Code available at: https://gist.github.com/qpwo/272df112928391b2c83a3b67732a5c25
# Author: Luke Harold Miles
# email: luke@cs.uky.edu
# Site: https://lukemiles.org

# Function: Cycle Finder
def simple_cycles(G):
    def _unblock(thisnode, blocked, B):
        stack = set([thisnode])
        while stack:
            node = stack.pop()
            if node in blocked:
                blocked.remove(node)
                stack.update(B[node])
                B[node].clear()
    G    = {v: set(nbrs) for (v,nbrs) in G.items()}
    sccs = strongly_connected_components(G)
    while sccs:
        scc       = sccs.pop()
        startnode = scc.pop()
        path      = [startnode]
        blocked   = set()
        closed    = set()
        blocked.add(startnode)
        B     = defaultdict(set)
        stack = [ (startnode,list(G[startnode])) ]
        while stack:
            thisnode, nbrs = stack[-1]
            if nbrs:
                nextnode = nbrs.pop()
                if nextnode == startnode:
                    yield path[:]
                    closed.update(path)
                elif nextnode not in blocked:
                    path.append(nextnode)
                    stack.append( (nextnode, list(G[nextnode])) )
                    closed.discard(nextnode)
                    blocked.add(nextnode)
                    continue
            if not nbrs:
                if thisnode in closed:
                    _unblock(thisnode, blocked, B)
                else:
                    for nbr in G[thisnode]:
                        if thisnode not in B[nbr]:
                            B[nbr].add(thisnode)
                stack.pop()
                path.pop()
        remove_node(G, startnode)
        H = subgraph(G, set(scc))
        sccs.extend(strongly_connected_components(H))

# Function: SCC       
def strongly_connected_components(graph):
    index_counter = [0]
    stack         = []
    lowlink       = {}
    index         = {}
    result        = []   
    def _strong_connect(node):
        index[node]   = index_counter[0]
        lowlink[node] = index_counter[0]
        index_counter[0] += 1
        stack.append(node) 
        successors = graph[node]
        for successor in successors:
            if successor not in index:
                _strong_connect(successor)
                lowlink[node] = min(lowlink[node],lowlink[successor])
            elif successor in stack:
                lowlink[node] = min(lowlink[node],index[successor])
        if lowlink[node] == index[node]:
            connected_component = []
            while True:
                successor = stack.pop()
                connected_component.append(successor)
                if successor == node: break
            result.append(connected_component[:])
    for node in graph:
        if node not in index:
            _strong_connect(node)
    return result

# Function: Remove Node
def remove_node(G, target):
    del G[target]
    for nbrs in G.values():
        nbrs.discard(target)

# Function: Subgraph
def subgraph(G, vertices):
    return {v: G[v] & vertices for v in vertices}

###############################################################################

# Function: Concordance Matrix
def concordance_matrix(dataset, W):
    concordance = np.zeros((dataset.shape[0], dataset.shape[0]))
    for i in range(0, concordance.shape[0]):
        for j in range(0, concordance.shape[1]):
            value = 0
            for k in range(0, dataset.shape[1]):
                if (dataset[i,k] >= dataset[j,k]):
                    value = value + W[k]
            concordance[i,j] = value      
    if (np.sum(W) != 0):
        concordance = concordance/np.sum(W)
    return concordance

# Function: Discordance Matrix
def discordance_matrix(dataset):
    delta       = np.max(np.amax(dataset, axis = 0) - np.amin(dataset, axis = 0))
    discordance = np.zeros((dataset.shape[0], dataset.shape[0]))
    for i in range(0, discordance.shape[0]):
        for j in range(0, discordance.shape[1]):
            discordance[i,j] = np.max((dataset[j,:]  - dataset[i,:]))/delta
            if (discordance[i,j] < 0):
                discordance[i,j] = 0    
    return discordance

# Function: Dominance Matrix
def dominance_matrix(concordance, discordance, c_hat = 0.75, d_hat = 0.50):
    dominance = np.zeros((concordance.shape[0], concordance.shape[0]))
    for i in range (0, dominance.shape[0]):
        for j in range (0, dominance.shape[1]):
            if (concordance[i,j] >= c_hat and discordance[i,j] <= d_hat and i != j):
                dominance[i, j] = 1                 
    return dominance

# Function: Find Cycles and Unites it as a Single Criteria
def johnson_algorithm_cycles(dominance):
    graph = {}
    value = [[] for i in range(dominance.shape[0])]
    keys  = range(dominance.shape[0])
    for i in range(0, dominance.shape[0]):
        for j in range(0, dominance.shape[0]):
            if (dominance[i,j] == 1):
                value[i].append(j)
    for i in keys:
        graph[i] = value[i]  
    s1 = list(simple_cycles(graph))
    for k in range(0, len(s1)):   
        for j in range(0, len(s1[k]) -1):
            dominance[s1[k][j], s1[k][j+1]] = 0
            dominance[s1[k][j+1], s1[k][j]] = 0
    s2 = s1[:]
    for m in s1:
        for n in s1:
            if set(m).issubset(set(n)) and m != n:
                s2.remove(m)
                break
        for i in range(0, dominance.shape[0]):
            count = 0
            for j in range(0, len(s2[k])):
                if (dominance[i, s2[k][j]] > 0):
                    count = count + 1
            if (count > 0):
                for j in range(0, len(s2[k])):
                    dominance[i, s2[k][j]] = 1
    return dominance

# Function: Electre I
def electre_i(dataset, W, remove_cycles = False, c_hat = 0.75, d_hat = 0.50, graph = True):
    kernel      = []
    dominated   = []
    concordance = concordance_matrix(dataset, W)
    discordance = discordance_matrix(dataset)
    dominance   = dominance_matrix(concordance, discordance, c_hat = c_hat, d_hat = d_hat)
    if (remove_cycles == True):
        dominance = johnson_algorithm_cycles(dominance)
    row_sum     = np.sum(dominance, axis = 0)
    kernel      = np.where(row_sum == 0)[0].tolist()            
    for j in range(0, dominance.shape[1]):
        for i in range(0, len(kernel)):
            if (dominance[kernel[i], j] == 1):
                if (j not in dominated):
                    dominated.append(j) 
    limit = len(kernel)
    for j in range(0, dominance.shape[1]):
        for i in range(0, limit):
            if (dominance[kernel[i], j] == 0 and np.sum(dominance[:,j], axis = 0) > 0):
                if (j not in dominated and j not in kernel):
                    kernel.append(j)
    kernel    = ['a' + str(alt + 1) for alt in kernel]
    dominated = ['a' + str(alt + 1) for alt in dominated]
    if (graph == True):
        for i in range(0, dominance.shape[0]):
            radius = 1
            node_x = radius*math.cos(math.pi * 2 * i / dominance.shape[0])
            node_y = radius*math.sin(math.pi * 2 * i / dominance.shape[0])
            if ('a' + str(i+1) in kernel):
                plt.text(node_x,  node_y, 'a' + str(i+1), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0.8, 1.0, 0.8),))
            else:
              plt.text(node_x,  node_y, 'a' + str(i+1), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (1.0, 0.8, 0.8),))  
        for i in range(0, dominance.shape[0]):
            for j in range(0, dominance.shape[1]):
                node_xi = radius*math.cos(math.pi * 2 * i / dominance.shape[0])
                node_yi = radius*math.sin(math.pi * 2 * i / dominance.shape[0])
                node_xj = radius*math.cos(math.pi * 2 * j / dominance.shape[0])
                node_yj = radius*math.sin(math.pi * 2 * j / dominance.shape[0])
                if (dominance[i, j] == 1):  
                    if ('a' + str(i+1) in kernel):
                        plt.arrow(node_xi, node_yi, node_xj - node_xi, node_yj - node_yi, head_width = 0.01, head_length = 0.2, overhang = 0.0, color = 'black', linewidth = 0.9, length_includes_head = True)
                    else:
                        plt.arrow(node_xi, node_yi, node_xj - node_xi, node_yj - node_yi, head_width = 0.01, head_length = 0.2, overhang = 0.0, color = 'red', linewidth = 0.9, length_includes_head = True)
        axes = plt.gca()
        axes.set_xlim([-radius, radius])
        axes.set_ylim([-radius, radius])
        plt.axis('off')
        plt.show() 
    return concordance, discordance, dominance, kernel, dominated

# Function: WASPAS
def waspas_method(dataset, criterion_type, weights, lambda_value):
    x = np.zeros((dataset.shape[0], dataset.shape[1]), dtype = float)
    for j in range(0, dataset.shape[1]):
        if (criterion_type[j] == 'max'):
            x[:,j] = 1 + ( dataset[:,j] - np.min(dataset[:,j]) ) / ( np.max(dataset[:,j]) - np.min(dataset[:,j]) )
        else:
            x[:,j] = 1 + ( np.max(dataset[:,j]) - dataset[:,j] ) / ( np.max(dataset[:,j]) - np.min(dataset[:,j]) )
    wsm    = np.sum(x*weights, axis = 1)
    wpm    = np.prod(x**weights, axis = 1)
    waspas = (lambda_value)*wsm + (1 - lambda_value)*wpm
    return wsm, wpm, waspas