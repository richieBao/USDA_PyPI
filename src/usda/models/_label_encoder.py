# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 15:14:55 2022

@author: richie bao
"""
from sklearn import preprocessing

def df_multiColumns_LabelEncoder(df,columns=None):    
    '''
    function - 根据指定的（多个）列，将分类转换为整数表示，区间为[0,分类数-1]
    
    Params:
        df - DataFrame格式数据；DataFrame
        columns - 指定待转换的列名列表；list(string)
        
    Returns:
        output - 分类整数编码；DataFrame
    '''    
    
    output=df.copy()
    if columns is not None:
        for col in columns:
            output[col]=preprocessing.LabelEncoder().fit_transform(output[col])
    else:
        for column_name, col in output.iteritems():
            output[column_name]=preprocessing.LabelEncoder().fit_transform(col)
            
    return output

