# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 21:30:06 2022

@author: richie bao
"""
import os

def filePath_extraction(dirpath,fileType): 
    '''
    funciton  - 以所在文件夹路径为键，值为包含该文件夹下所有文件名的列表。文件类型可以自行定义 
    
    Params:
        dirpath - 根目录，存储所有待读取的文件；string
        fileType - 待读取文件的类型；list(string)
        
    Returns:
        filePath_Info - 文件路径字典，文件夹路径为键，文件夹下的文件名列表为值；dict
    '''    
    
    filePath_Info={}
    i=0
    for dirpath,dirNames,fileNames in os.walk(dirpath): # os.walk()遍历目录，使用help(os.walk)查看返回值解释
        i+=1
        if fileNames:  # 仅当文件夹中有文件时才提取
            tempList=[f for f in fileNames if f.split('.')[-1] in fileType]
            if tempList:  # 剔除文件名列表为空的情况，即文件夹下存在不为指定文件类型的文件时，上一步列表会返回空列表[]
                filePath_Info.setdefault(dirpath,tempList)
    return filePath_Info

