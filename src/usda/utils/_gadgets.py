# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 08:25:20 2022

@author: richie bao
"""
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import torch
import pandas as pd
import matplotlib

def variable_name(var):
    '''
    function - 将变量名转换为字符串
    
    Parasm:
        var - 变量名
    '''
    
    return [tpl[0] for tpl in filter(lambda x: var is x[1], globals().items())][0]


def lst_index_split(lst, args):
    '''
    function - 根据索引，分割列表
    
    transfer:https://codereview.stackexchange.com/questions/47868/splitting-a-list-by-indexes/47877 
    '''
    if args:
        args=(0,) + tuple(data+1 for data in args) + (len(lst)+1,)
    seg_list=[]
    for start, end in zip(args, args[1:]):
        seg_list.append(lst[start:end])
        
    return seg_list

flatten_lst=lambda lst: [m for n_lst in lst for m in flatten_lst(n_lst)] if type(lst) is list else [lst]

def nestedlst_insert(nestedlst):
    '''
    function - 嵌套列表，子列表前后插值
    
    Params:
        nestedlst - 嵌套列表；list
    
    Returns:
        nestedlst - 分割后的列表；list
    '''
    for idx in range(len(nestedlst)-1):
        nestedlst[idx+1].insert(0,nestedlst[idx][-1])
    nestedlst.insert(0,nestedlst[0])
    
    return nestedlst

    
class AttrDict(dict):
    """
    # Code adapted from:
    # https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/collections.py

    Source License
    # Copyright (c) 2017-present, Facebook, Inc.
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    ##############################################################################
    #
    # Based on:
    # --------------------------------------------------------
    # Fast R-CNN
    # Copyright (c) 2015 Microsoft
    # Licensed under The MIT License [see LICENSE for details]
    # Written by Ross Girshick
    # --------------------------------------------------------
    """
    IMMUTABLE = '__immutable__'

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__[AttrDict.IMMUTABLE] = False

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if not self.__dict__[AttrDict.IMMUTABLE]:
            if name in self.__dict__:
                self.__dict__[name] = value
            else:
                self[name] = value
        else:
            raise AttributeError(
                'Attempted to set "{}" to "{}", but AttrDict is immutable'.
                format(name, value)
            )

    def immutable(self, is_immutable):
        """Set immutability to is_immutable and recursively apply the setting
        to all nested AttrDicts.
        """
        self.__dict__[AttrDict.IMMUTABLE] = is_immutable
        # Recursively set immutable state
        for v in self.__dict__.values():
            if isinstance(v, AttrDict):
                v.immutable(is_immutable)
        for v in self.values():
            if isinstance(v, AttrDict):
                v.immutable(is_immutable)

    def is_immutable(self):
        return self.__dict__[AttrDict.IMMUTABLE]
    
def sequence2series_of_overlapping_with_labels(seq,ws):
    out=[]
    length=len(seq)
    for i in range(length-ws):
        win=seq[i:i+ws]
        label=seq[i+ws:i+ws+1]
        out.append((win,label))

    return out    

def create_sequence2series_datasetNdataloader_from_df(df,cols,window_size=10,test_size_ratio=0.1,batch_size=64):
    test_size=int(len(df)*test_size_ratio)
    train_set,test_set=df[:-test_size][cols],df[-test_size:][cols]
    
    train_data_dict={}
    train_loader_dict={}
    test_data_dict={}
    test_loader_dict={}
    
    def ds_dl(ds_df,col):
        s2s=sequence2series_of_overlapping_with_labels(ds_df[col].values, window_size)
        X,y=zip(*s2s)
        X=np.vstack(X)
        y=np.hstack(y)
        X,y=torch.Tensor(X),torch.Tensor(y) #.type(torch.LongTensor) 
        loader=data.DataLoader(data.TensorDataset(X,y), shuffle=True, batch_size=batch_size)
        return (X,y),loader 
    
    for col in cols:
        train_data,train_loader=ds_dl(train_set,col)
        train_data_dict[col]=train_data
        train_loader_dict[col]=train_loader
        
        test_data,test_loader=ds_dl(test_set,col)
        test_data_dict[col]=test_data
        test_loader_dict[col]=test_loader        
        
    return train_data_dict,test_data_dict,train_loader_dict,test_loader_dict   

def normalize_by_meanNstd(df):
    mean_std_dict={col:[df[col].mean(),df[col].std()] for col in df.columns}
    #print(mean_std_dict)
    norm_dict={}
    for col in df.columns:
        norm_dict[col]=(df[col]-mean_std_dict[col][0])/mean_std_dict[col][1]
    norm_df=pd.DataFrame(norm_dict)
    return norm_df,mean_std_dict

def normalize_by_minmax4all(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))

def cmap2hex(cmap_name,N): 
    cmap = matplotlib.cm.get_cmap(cmap_name, N)
    hex_colors={i:matplotlib.colors.rgb2hex(cmap(i)) for i in range(N)}
    return hex_colors 

if __name__=="__main__":
    xyz=6
    var_str=variable_name(xyz)
    print(type(var_str))