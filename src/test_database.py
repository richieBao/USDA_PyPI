# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 08:49:18 2022

@author: richie bao
"""
import usda
from usda import *
from pathlib import Path
# print(dir(datasets))

def test_load_sales_data_cartoon_database():
    data=datasets.load_sales_data_cartoon_database()
    print(data)
    sales_table=data.sales_table
    print(sales_table)

def test_df2SQLite():
    data=datasets.load_sales_data_cartoon_database()
    sales_table=data.sales_table    
    db_fp="./test/database/fruits.sqlite"
    database.df2SQLite(db_fp,sales_table,"sales")  
    print(help(database.df2SQLite))
    
def test_SQLite2df():
    db_fp="./test/database/fruits.sqlite"
    data=database.SQLite2df(db_fp,"sales")
    print(data)

def test_postSQL2gpd():
    AoT_nodes_gdf=database.postSQL2gpd(table_name='AoT_nodes',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='AoT')
    print(AoT_nodes_gdf)
    
def test_DisplayablePath():  
    app_root=r'C:\Users\richi\Pictures'    
    paths = utils.DisplayablePath.make_tree(Path(app_root))
    for path in paths:
        print(path.displayable())    

if __name__=="__main__":
    print(dir(usda))
    print("-"*50)
    # test_load_sales_data_cartoon_database()
    # test_df2SQLite()
    # test_SQLite2df()
    # test_postSQL2gpd()
    test_DisplayablePath()
    
    
