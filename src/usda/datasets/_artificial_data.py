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
        
def ramen_price_cartoon_statistic():
    ramen_price=pd.DataFrame([700,850,600,650,980,750,500,890,880,700,890,720,680,650,790,670,680,900,880,720,850,700,780,850,750,
                              780,590,650,580,750,800,550,750,700,600,800,800,880,790,790,780,600,690,680,650,890,930,650,777,700],columns=["price"])
    
    with open("./data/ramen_price_cartoon_statistic.pickle",'wb') as f:
        pickle.dump(ramen_price, f,pickle.HIGHEST_PROTOCOL)        
        print("dumped ramen_price_cartoon_statistic.")
        
def bowling_contest_cartoon_statistic():
    bowlingContest_scores_dic={'A_team':{'Barney':86,'Harold':73,'Chris':124,'Neil':111,'Tony':90,'Simon':38},
                               'B_team':{'Jo':84,'Dina':71,'Graham':103,'Joe':85,'Alan':90,'Billy':89},
                               'C_team':{'Gordon':229,'Wade':77,'Cliff':59,'Arthur':95,'David':70,'Charles':88}
                               }    
    
    bowlingContest_scores=pd.DataFrame.from_dict(bowlingContest_scores_dic, orient='index').stack().to_frame(name='score')          
    with open("./data/bowling_contest_cartoon_statistic.pickle",'wb') as f:
        pickle.dump(bowlingContest_scores, f,pickle.HIGHEST_PROTOCOL)        
        print("dumped bowling_contest_cartoon_statistic.")    
        
def test_score_cartoon_statistic():
    test_score_dic={"English":{"Mason":90,"Reece":81,'A':73,'B':97,'C':85,'D':60,'E':74,'F':64,'G':72,'H':67,'I':87,'J':78,'K':85,'L':96,'M':77,'N':100,'O':92,'P':86},
                    "Chinese":{"Mason":71,"Reece":90,'A':79,'B':70,'C':67,'D':66,'E':60,'F':83,'G':57,'H':85,'I':93,'J':89,'K':78,'L':74,'M':65,'N':78,'O':53,'P':80},
                    "history":{"Mason":73,"Reece":61,'A':74,'B':47,'C':49,'D':87,'E':69,'F':65,'G':36,'H':7,'I':53,'J':100,'K':57,'L':45,'M':56,'N':34,'O':37,'P':70},
                    "biology":{"Mason":59,"Reece":73,'A':47,'B':38,'C':63,'D':56,'E':75,'F':53,'G':80,'H':50,'I':41,'J':62,'K':44,'L':26,'M':91,'N':35,'O':53,'P':68},
                   }
    
    test_score=pd.DataFrame.from_dict(test_score_dic)    
    with open("./data/test_score_cartoon_statistic.pickle",'wb') as f:
        pickle.dump(test_score,f,pickle.HIGHEST_PROTOCOL)        
        print("dumped test_score_cartoon_statistic.")          
        
        
if __name__=="__main__":
    print("-"*50)
    # sales_data_cartoon_database()
    # ramen_price_cartoon_statistic()
    # bowling_contest_cartoon_statistic()
    test_score_cartoon_statistic()