# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 19:00:53 2022

@author: richie bao
"""
from sqlalchemy import create_engine,text
import pandas as pd
import geopandas as gpd

def df2SQLite(db_fp, df, table_name, method='fail'):    
    '''
    function - 把pandas DataFrame格式数据写入数据库（同时创建表）
    
    Paras:
        db_fp - 数据库链接；string
        df - 待写入数据库的DataFrame格式数据；DataFrame
        table - 表名称；string
        method - 写入方法，'fail'，'replace'或'append'；string
    Returns:
        None    
    '''    
    
    engine=create_engine('sqlite:///'+'\\\\'.join(db_fp.split('\\')),echo=True) 
    try:    
        df.to_sql(table_name, con=engine, if_exists="%s" % method)
        if method=='replace':            
            print("_"*10,'the %s table has been overwritten...'%table_name)                  
        elif method=='append':
            print("_"*10, 'the %s table has been appended...' % table_name)
        else:
            print("_"*10, 'the %s table has been written......' % table_name)
    except:
        print("_"*10, 'the %s table has been existed......' % table_name)
        
def SQLite2df(db_fp,table):   
    '''
    function - pandas方法，从SQLite数据库中读取表数据
    
    Paras:
        db_fp - 数据库文件路径；string
        table - 所要读取的表；string

    Returns:
        读取的表；DataFrame        
    '''    
    
    return pd.read_sql_table(table, 'sqlite:///'+'\\\\'.join(db_fp.split('\\'))) #pd.read_sql_table从数据库中读取指定的表        
        
def gpd2postSQL(gdf,table_name,**kwargs):   
    '''
    function - 将GeoDataFrame格式数据写入PostgreSQL数据库
    
    Paras:
        gdf - GeoDataFrame格式数据，含geometry字段（几何对象，点、线和面，数据值对应定义的坐标系统）；GeoDataFrame
        table_name - 写入数据库中的表名；string
        **kwargs - 连接数据库相关信息，包括myusername（数据库的用户名），mypassword（用户密钥），mydatabase（数据库名）；string
        
    Returns:
        None
    '''     
    
    engine=create_engine("postgresql://{myusername}:{mypassword}@localhost:5432/{mydatabase}".format(myusername=kwargs['myusername'],mypassword=kwargs['mypassword'],mydatabase=kwargs['mydatabase']))  
    gdf.to_postgis(table_name, con=engine, if_exists='replace', index=False,)  
    print("_"*50)
    print('The GeoDataFrame has been written to the PostgreSQL database.The table name is {}.'.format(table_name))

def postSQL2gpd(table_name,geom_col='geometry',**kwargs):    
    '''
    function - 读取PostgreSQL数据库中的表为GeoDataFrame格式数据
    
    Paras:
        table_name - 待读取数据库中的表名；string
        geom_col='geometry' - 几何对象，常规默认字段为'geometry'；string
        **kwargs - 连接数据库相关信息，包括myusername（数据库的用户名），mypassword（用户密钥），mydatabase（数据库名）；string
    Returns:
        读取的表数据；GeoDataFrame
    '''

    engine=create_engine("postgresql://{myusername}:{mypassword}@localhost:5432/{mydatabase}".format(myusername=kwargs['myusername'],mypassword=kwargs['mypassword'],mydatabase=kwargs['mydatabase']))  
    sql=f"SELECT * FROM {table_name}"
    gdf=gpd.read_postgis(sql=text(sql), con=engine.connect(),geom_col=geom_col)
    print("_"*50)
    print('The data has been read from PostgreSQL database. The table name is {}.'.format(table_name))    
    return gdf          
                
def df2postSQL(df,table_name,if_exists='replace',**kwargs):
    '''
    function - 将DataFrame格式数据写入PostgreSQL数据库
    
    Paras:
        df - DataFrame格式数据
        table_name - 写入数据库中的表名
        **kwargs - 连接数据库相关信息，包括myusername（数据库的用户名），mypassword（用户密钥），mydatabase（数据库名）
    '''    
    #The URI should start with postgresql:// instead of postgres://. SQLAlchemy used to accept both, but has removed support for the postgres name.
    engine=create_engine("postgresql://{myusername}:{mypassword}@localhost:5432/{mydatabase}".format(myusername=kwargs['myusername'],mypassword=kwargs['mypassword'],mydatabase=kwargs['mydatabase']))  
    conn=engine.connect()
    df.to_sql(table_name, con=conn, if_exists=if_exists,index=False)    
    # gdf.to_postgis(table_name, con=engine, if_exists='replace', index=False,)  
    print("_"*50)    
    print('The GeoDataFrame has been written to the PostgreSQL database.The table name is {}.'.format(table_name))   

def postSQL2df(table_name,**kwargs):    
    '''
    function - 读取PostgreSQL数据库中的表为DataFrame格式数据
    
    Paras:
        table_name - 待读取数据库中的表名
        **kwargs - 连接数据库相关信息，包括myusername（数据库的用户名），mypassword（用户密钥），mydatabase（数据库名）
    '''    
    engine=create_engine("postgresql://{myusername}:{mypassword}@localhost:5432/{mydatabase}".format(myusername=kwargs['myusername'],mypassword=kwargs['mypassword'],mydatabase=kwargs['mydatabase']))  
    conn=engine.connect()
    df=pd.read_sql('SELECT * FROM {}'.format(table_name), conn)

    print("_"*50)
    print('The data has been read from PostgreSQL database. The table name is {}.'.format(table_name))    
    return df 