# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 15:31:39 2022

@author: richie bao
"""
import math
import pandas as pd  

def entropy_compomnent(numerator,denominator):    
    '''
    function - 计算信息熵分量
    
    Params:
        numerator - 分子；
        denominator - 分母；
        
    Returns:
        信息熵分量；float
    '''    
    
    if numerator!=0:
        return -numerator/denominator*math.log2(numerator/denominator)    
    elif numerator==0:
        return 0
    
def IG(df_dummies):      
    '''
    function - 计算信息增量（IG）
    
    Params:
        df_dummies - DataFrame格式，独热编码的特征值；DataFrame
        
    Returns:
        cal_info_df - 信息增益（Information gain）；DataFrame
    '''    
    
    weighted_frequency=df_dummies.apply(pd.Series.value_counts)
    weighted_sum=weighted_frequency.sum(axis=0) 
    feature_columns=weighted_frequency.columns.tolist()
    Parent_entropy=entropy_compomnent(weighted_frequency[feature_columns[-1]][0],14)+entropy_compomnent(weighted_frequency[feature_columns[-1]][1],14)
    
    cal_info=[]
    for feature in feature_columns[:-2]:        
        v_0_frequency=df_dummies.query('%s==0'%feature).iloc[:,-1].value_counts().reindex(df_dummies[feature].unique(),fill_value=0) #频数可能为0，如果为0则会被舍弃（value_counts），因此需要补回（.reindex）
        v_1_frequency=df_dummies.query('%s==1'%feature).iloc[:,-1].value_counts().reindex(df_dummies[feature].unique(),fill_value=0)
        first_child_entropy=entropy_compomnent(v_0_frequency[0], v_0_frequency.sum(axis=0))+entropy_compomnent(v_0_frequency[1], v_0_frequency.sum(axis=0)) 
        second_child_entropy=entropy_compomnent(v_1_frequency[0], v_1_frequency.sum(axis=0))+entropy_compomnent(v_1_frequency[1], v_1_frequency.sum(axis=0))

        cal_dic={'test':feature,
                 'Parent_entropu':Parent_entropy,
                 'first_child_entropy':first_child_entropy,
                 'second_child_entropy':second_child_entropy,
                 'Weighted_average_expression':'%f*%d/%d+%f*%d/%d'%(first_child_entropy,weighted_frequency[feature][0],weighted_sum.loc[feature],second_child_entropy,weighted_frequency[feature][1],weighted_sum.loc[feature]),
                 'IG':first_child_entropy*(weighted_frequency[feature][0]/weighted_sum.loc[feature])+second_child_entropy*(weighted_frequency[feature][1]/weighted_sum.loc[feature])
                 
                }
        cal_info.append(cal_dic)
    cal_info_df=pd.DataFrame.from_dict(cal_info)    
    return cal_info_df    