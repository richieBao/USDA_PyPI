# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 15:05:33 2022

@author: richie bao
"""
import pandas as pd
import util_misc
import os  

def KITTI_info(KITTI_info_fp,timestamps_fp):
    '''
    function - 读取KITTI文件信息，1-包括经纬度，惯性导航系统信息等的.txt文件，2-包含时间戳的.txt文件
    
    Params:
        KITTI_info_fp - 数据文件路径；string
        timestamps_fp - 时间戳文件路径；string
        
    Returns:
        drive_info - 返回数据；DataFrame
    '''  

    drive_fp=util_misc.filePath_extraction(KITTI_info_fp,['txt'])
    '''展平列表函数'''
    flatten_lst=lambda lst: [m for n_lst in lst for m in flatten_lst(n_lst)] if type(lst) is list else [lst]
    drive_fp_list=flatten_lst([[os.path.join(k,f) for f in drive_fp[k]] for k,v in drive_fp.items()])

    columns=["lat","lon","alt","roll","pitch","yaw","vn","ve","vf","vl","vu","ax","ay","ay","af","al","au","wx","wy","wz","wf","wl","wu","pos_accuracy","vel_accuracy","navstat","numsats","posmode","velmode","orimode"]
    drive_info=pd.concat([pd.read_csv(item,delimiter=' ',header=None) for item in drive_fp_list],axis=0)
    drive_info.columns=columns
    drive_info=drive_info.reset_index()
    
    timestamps=pd.read_csv(timestamps_fp,header=None)
    timestamps.columns=['timestamps_']
    drive_info=pd.concat([drive_info,timestamps],axis=1,sort=False)
    #drive_29_0071_info.index=pd.to_datetime(drive_29_0071_info["timestamps_"]) #用时间戳作为行(row)索引
    return drive_info

