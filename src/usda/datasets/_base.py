# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 19:58:26 2022

@author: richie bao
"""
import pickle
import os
from importlib import resources
from ..utils import Bunch

# DATA_MODULE="usda.datasets.data"
DATA_MODULE="usda.datasets.data"

def load_sales_data_cartoon_database(data_module=DATA_MODULE):
    '''
    测试用数据。根据《漫画数据库》一书中的销售数据集构建

    Parameters
    ----------
    data_module : TYPE, string
        数据所在文件夹. The default is DATA_MODULE.

    Returns
    -------
    Type, Class
        假数据：销售数据.含属性（字段）：'sales_table','exporting_country_table','sale_details_table','commodity_table'

    '''
    data_file_name="sales_data_cartoon_database.pickle" 
    with resources.open_binary(data_module,data_file_name) as data_file:
        data=pickle.load(data_file)
    sales_table=data['sales_table']
    exporting_country_table=data['exporting_country_table']
    sale_details_table=data['sale_details_table']
    commodity_table=data['commodity_table']
    
    table_names=['sales_table','exporting_country_table','sale_details_table','commodity_table']   
    
    return Bunch(sales_table=sales_table,
                 exporting_country_table=exporting_country_table，
                 sale_details_table=sale_details_table，
                 commodity_table=commodity_table，
                 file_name=data_file_name)




if __name__=="__main__":
    sales_data_cartoon_databas=load_sales_data_cartoon_database()
    
