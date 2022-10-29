# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 09:36:35 2022

@author: richie bao
"""
import os,re

def LandsatMTL_info(fp):
    '''
    function - 读取landsat *_MTL.txt文件，提取需要的信息
    
    Paras:
        fp - Landsat 文件根目录；string
    
    return:
        band_fp_dic - 返回各个波段的路径字典；dict
        Landsat_para - 返回Landsat 参数 ；dict
    '''    
    
    fps=[os.path.join(root,file) for root, dirs, files in os.walk(fp) for file in files] # 提取文件夹下所有文件的路径
    MTLPattern=re.compile(r'_MTL.txt',re.S) # 匹配对象模式，提取_MTL.txt遥感影像的元数据文件
    MTLFn=[fn for fn in fps if re.findall(MTLPattern,fn)][0]
    with open(MTLFn,'r') as f: # 读取所有元数据文件信息
        MTLText=f.read()
    bandFn_Pattern=re.compile(r'FILE_NAME_BAND_[0-9]\d* = "(.*?)"\n',re.S)  # Landsat 波段文件
    band_fn=re.findall(bandFn_Pattern,MTLText)
    band_fp=[[(re.findall(r'B[0-9]\d*',fn)[0], re.findall(r'.*?%s$'%fn,f)[0]) for f in fps if re.findall(r'.*?%s$'%fn,f)] for fn in band_fn] # (文件名，文件路径)
    band_fp_dic={i[0][0]:i[0][1] for i in band_fp}
    # 需要数据的提取标签/根据需要读取元数据信息
    values_fields=["RADIANCE_ADD_BAND_10",
                   "RADIANCE_ADD_BAND_11",
                   "RADIANCE_MULT_BAND_10",
                   "RADIANCE_MULT_BAND_11",
                   "K1_CONSTANT_BAND_10",
                   "K2_CONSTANT_BAND_10",
                   "K1_CONSTANT_BAND_11",
                   "K2_CONSTANT_BAND_11",
                   "DATE_ACQUIRED",
                   "SCENE_CENTER_TIME",
                   "MAP_PROJECTION",
                   "DATUM",
                   "UTM_ZONE"]

    Landsat_para={field:re.findall(re.compile(r'%s = "*(.*?)"*\n'%field),MTLText)[0] for field in values_fields} #（参数名，参数值）
    return  band_fp_dic,Landsat_para # 返回所有波段路径和需要的参数值

