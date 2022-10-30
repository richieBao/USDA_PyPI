# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 20:12:55 2022

@author: richie bao
"""
import time
import os
import json

def save_as_json(array,save_root,fn):    
    '''
    function - 保存文件,将文件存储为json数据格式
    
    Params:
        array - 待保存的数组；array
        save_root - 文件保存的根目录 ；string
        fn - 保存的文件名；string
        
    Returns:
        None
    '''    
    
    json_file=open(os.path.join(save_root,r'%s_'%fn+str(time.time()))+'.json','w')
    json.dump(array.tolist(),json_file)  # 将numpy数组转换为列表后存储为json数据格式
    json_file.close()
    
    
    