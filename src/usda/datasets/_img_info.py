# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 19:11:05 2022

@author: richie bao
"""
from PIL import Image
from PIL.ExifTags import TAGS
from datetime import datetime
import time   

def img_exif_info(img_fp,printing=True):
    '''
    function - 提取数码照片的属性信息和拍摄数据，即可交换图像文件格式（Exchangeable image file format，Exif）
    
    Params:
        img_fp - 一个图像的文件路径；string
        printing - 是否打印。The default is True；bool
        
    Returns:
        exif_ - 提取的照片信息结果；dict
    ''' 
    
    img=Image.open(img_fp,)
    try:
        img_exif=img._getexif()
        exif_={TAGS[k]: v for k, v in img_exif.items() if k in TAGS}  
        # 由2017:07:20 09:16:58格式时间，转换为时间戳，用于比较时间先后。
        time_lst=[int(i) for i in re.split(' |:',exif_['DateTimeOriginal'])] # DateTimeOriginal;'DateTime'
        time_tuple=datetime.timetuple(datetime(time_lst[0], time_lst[1], time_lst[2], time_lst[3], time_lst[4], time_lst[5],))
        time_stamp=time.mktime(time_tuple)
        exif_["timestamp"]=time_stamp
        
    except ValueError:
        print("exif not found!")
    for tag_id in img_exif: # 取Exif信息 iterating over all EXIF data fields
        tag=TAGS.get(tag_id,tag_id) # 获取标签名 get the tag name, instead of human unreadable tag id
        data=img_exif.get(tag_id)
        if isinstance(data,bytes): # 解码 decode bytes 
            try:
                data=data.decode()
            except ValueError:
                data="tag:%s data not found."%tag
        if printing:   
            print(f"{tag:30}:{data}")

    # 将度转换为浮点数，Decimal Degrees = Degrees + minutes/60 + seconds/3600
    if 'GPSInfo' in exif_:   
        GPSInfo=exif_['GPSInfo']
        geo_coodinate={
            "GPSLatitude":float(GPSInfo[2][0]+GPSInfo[2][1]/60+GPSInfo[2][2]/3600),
            "GPSLongitude":float(GPSInfo[4][0]+GPSInfo[4][1]/60+GPSInfo[4][2]/3600),
            "GPSAltitude":GPSInfo[6],
            "GPSTimeStamp_str":"%d:%f:%f"%(GPSInfo[7][0],GPSInfo[7][1]/10,GPSInfo[7][2]/100),#字符形式
            "GPSTimeStamp":float(GPSInfo[7][0]+GPSInfo[7][1]/10+GPSInfo[7][2]/100),#浮点形式
            "GPSImgDirection":GPSInfo[17],
            "GPSDestBearing":GPSInfo[24],
            "GPSDateStamp":GPSInfo[29],
            "GPSHPositioningError":GPSInfo[31],            
        }    
        if printing: 
            print("_"*50)
            print(geo_coodinate)
        return exif_,geo_coodinate
    else:
        return exif_