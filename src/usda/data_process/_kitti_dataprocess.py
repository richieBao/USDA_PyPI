# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 13:48:15 2022

@author: richie bao
"""
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine

def KITTI_info2sqlite(imgsPath_fp,info_fp,replace_path,db_fp,table,method='fail'):  
    '''
    function - 将KITTI图像路径与经纬度信息对应起来，并存入SQLite数据库
    
    Params:
        imgsPath_fp - 图像文件路径；string
        info_fp - 图像信息文件路径；string
        replace_path - 替换路径名；string
        db_fp - SQLite数据库路径；string
        table - 数据库表名；string
        method - 包括fail, replace, append等。The default is'fail'；string    

    Returns:
        None
    '''    
    
    imgsPath=pd.read_pickle(imgsPath_fp)
    # flask Jinja的url_for仅支持'/,因此需要替换'\\'
    imgsPath_replace=imgsPath.imgs_fp.apply(lambda row:str(Path(replace_path).joinpath(Path(row).name)).replace('\\','/'))
    info=pd.read_pickle(info_fp)
    imgs_df=pd.concat([imgsPath_replace,info],axis=1)   
    engine=create_engine('sqlite:///'+'\\\\'.join(db_fp.split('\\')),echo=True)     

    try:
        imgs_df.to_sql('%s'%table,con=engine,index=False,if_exists="%s"%method)
        print("if_exists=%s:------Data has been written to the database!"%method)
    except:
        print("_"*15,'the %s table has been existed...'%table)