# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 15:14:55 2022

@author: richie bao
"""
from sklearn import preprocessing
from sklearn.utils import Bunch
import torch

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

def text2ints_encoded(filename,seq_length=100):

    # load ascii text and covert to lowercase
    raw_text=open(filename, 'r', encoding='utf-8').read()
    raw_text=raw_text.lower()

    # create mapping of unique chars to integers
    chars=sorted(list(set(raw_text)))
    char_to_int=dict((c, i) for i, c in enumerate(chars))

    # summarize the loaded data
    n_chars = len(raw_text)
    n_vocab = len(chars)
    print("Total Characters: ", n_chars)
    print("Total Vocab: ", n_vocab)    

    dataX = []
    dataY = []
    for i in range(0, n_chars - seq_length, 1):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
    n_patterns = len(dataX)
    print("Total Patterns: ", n_patterns)    
            
    # reshape X to be [samples, time steps, features]
    X = torch.tensor(dataX, dtype=torch.float32).reshape(n_patterns, seq_length, 1)
    X = X / float(n_vocab)
    y = torch.tensor(dataY)        
    
    return Bunch(data=X,target=y,feature_names=chars,char_to_int=char_to_int)

