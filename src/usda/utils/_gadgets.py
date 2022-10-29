# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 08:25:20 2022

@author: richi
"""

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

if __name__=="__main__":
    xyz=6
    var_str=variable_name(xyz)
    print(type(var_str))