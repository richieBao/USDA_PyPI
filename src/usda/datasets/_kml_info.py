# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 16:56:48 2022

@author: richie bao
"""

def kml_coordiExtraction(kml_pathDict):   
    '''
    function - 提取.kml文件中的坐标信息
    
    Params:
        kml_pathDict - .kml文件路径字典。文件夹名为键，值为包含该文件夹下所有文件名的列表。使用filePath_extraction()函数提取。
    
    Returns:
        kml_coordi_dict - 返回坐标信息；dict
    '''
    
    kml_CoordiInfo={}
    '''正则表达式函数，将字符串转换为模式对象.号匹配除换行符之外的任何字符串，但只匹配一个字母，增加*？字符代表匹配前面表达式的0个或多个副本，并匹配尽可能少的副本'''
    pattern_coodi=re.compile('<coordinates>(.*?)</coordinates>') 
    pattern_name=re.compile('<name>(.*?)</name>')
    count=0
    kml_coordi_dict={}
    for key in kml_pathDict.keys():
        temp_dict={}
        for val in kml_pathDict[key]:
            f=open(os.path.join(key,val),'r',encoding='UTF-8') # .kml文件中含有中文
            content=f.read().replace('\n',' ') # 移除换行，从而可以根据模式对象提取标识符间的内容，同时忽略换行
            name_info=pattern_name.findall(content)
            coordi_info=pattern_coodi.findall(content)
            coordi_info_processing=[coordi.strip(' ').split('\t\t') for coordi in coordi_info]
            print("名称数量：%d,坐标列表数量：%d"%(len(name_info),len(coordi_info_processing))) # 名称中包含了文件名<name>default_20170720081441</name>和文件夹名<name>线路标记点</name>。位于文头。
            name_info_id=[name_info[2:][n]+'_ID_'+str(n) for n in range(len(name_info[2:]))] # 名称有重名，用ID标识
            name_coordi=dict(zip(name_info_id,coordi_info_processing)) 
            for k in name_coordi.keys():                
                temp=[]
                for coordi in name_coordi[k]:
                    coordi_split=coordi.split(',')
                    # 提取的坐标值字符，可能不正确，不能转换为浮点数，因此通过异常处理
                    try:  
                        one_coordi=[float(i) for i in coordi_split]                        
                        if len(one_coordi)==3:# 可能提取的坐标值除了经纬度和高程，会出现多余或者少于3的情况，判断后将其忽略
                            temp.append(one_coordi)
                    except ValueError:
                        count=+1
                temp_dict[k]=temp
        
            kml_coordi_dict[os.path.join(key,val)]=temp_dict
            print("kml_坐标字典键：",kml_coordi_dict.keys())       
    f.close()
    return kml_coordi_dict
