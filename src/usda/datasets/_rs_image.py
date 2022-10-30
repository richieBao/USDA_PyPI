# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 15:00:52 2022

@author: richie bao
"""
import xml.etree.ElementTree as ET


def Sentinel2_bandFNs(MTD_MSIL2A_fn):    
    '''
    funciton - 获取sentinel-2波段文件路径，和打印主要信息
    
    Params:
        MTD_MSIL2A_fn - MTD_MSIL2A 文件路径；string
    
    Returns:
        band_fns_list - 波段相对路径列表；list(string)
        band_fns_dict - 波段路径为值，反应波段信息的字段为键的字典；dict
    '''    
    
    Sentinel2_tree=ET.parse(MTD_MSIL2A_fn)
    Sentinel2_root=Sentinel2_tree.getroot()

    print("GENERATION_TIME:{}\nPRODUCT_TYPE:{}\nPROCESSING_LEVEL:{}".format(Sentinel2_root[0][0].find('GENERATION_TIME').text,
                                                           Sentinel2_root[0][0].find('PRODUCT_TYPE').text,                 
                                                           Sentinel2_root[0][0].find('PROCESSING_LEVEL').text
                                                          ))    
    print("MTD_MSIL2A.xml 文件父结构:")
    for child in Sentinel2_root:
        print(child.tag,"-",child.attrib)
    print("_"*50)    
    band_fns_list=[elem.text for elem in Sentinel2_root.iter('IMAGE_FILE')] #[elem.text for elem in Sentinel2_root[0][0][11][0][0].iter()]
    band_fns_dict={f.split('_')[-2]+'_'+f.split('_')[-1]:f+'.jp2' for f in band_fns_list}
    print('获取sentinel-2波段文件路径:\n',band_fns_dict)
    
    return band_fns_list,band_fns_dict