# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 20:11:47 2022

@author: richie bao
"""
import pandas as pd
from datetime import datetime
import pickle

def sales_data_cartoon_database():    
    '''
    根据*漫画数据库*中的销售数据集录入数据
    该数据包括4个表，分别是销售表`sales_table`，包含的字段（列）有报表编码`idx`，日期`date`和出口国编码`exporting_country_ID`；出口国表`exporting_country_table`，包含的字段有出口国编码`exporting_country_ID`和出口国名称`exporting_country_name`；销售明细表`sale_details_table`，包含的字段有报表编码`idx`，商品编码`commodity_code`和数量`number`；商品表`commodity_table`，包含的字段有商品编码`commodity_code`和商品名称`commodity_name`。

    Returns
    -------
    None.

    '''
    # 定义字典类型的假设数据
    sales_dic={'idx':[1101,1102,1103,1104,1105],
               'date':[datetime(2020,3,5),datetime(2020,3,7),datetime(2020,3,8),datetime(2020,3,10),datetime(2020,3,12)],
               "exporting_country_ID":[12,23,25,12,25]}
    exporting_country_dic={"exporting_country_ID":[12,23,25],
                           'exporting_country_name':['kenya','brazil','peru']}
    sale_details_dic={'idx':[1101,1101,1102,1103,1104,1105,1105],
                      'commodity_code':[101,102,103,104,101,103,104],
                      'number':[1100,300,1700,500,2500,2000,700]}
    commodity_dic={'commodity_code':[101,102,103,104],
                   'commodity_name':['muskmelon','strawberry','apple','lemon']}
    
    # 为方便数据管理，将字典格式数据转换为pandas的DataFrame格式
    sales_table=pd.DataFrame.from_dict(sales_dic)
    exporting_country_table=pd.DataFrame.from_dict(exporting_country_dic)
    sale_details_table=pd.DataFrame.from_dict(sale_details_dic)
    commodity_table=pd.DataFrame.from_dict(commodity_dic)    

    sales_data_cartoon_database={"sales_table":sales_table,
                                 "exporting_country_table":exporting_country_table,
                                 "sale_details_table":sale_details_table,
                                 "commodity_table":commodity_table}
    with open("./data/sales_data_cartoon_database.pickle",'wb') as f:
        pickle.dump(sales_data_cartoon_database, f,pickle.HIGHEST_PROTOCOL)        
        print("dumped sales_data_cartoon_database.")


if __name__=="__main__":
    sales_data_cartoon_database()

