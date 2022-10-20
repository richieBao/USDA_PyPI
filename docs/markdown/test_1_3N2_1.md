> 测试部分按照[城市空间数据分析方法-过程稿](https://richiebao.github.io/USDA_CH_final/#/)章节测试。

## 1.3，2.1章测试

```python

import usda
from usda import *
from pathlib import Path
import pandas as pd
# print(dir(datasets))

def test_load_sales_data_cartoon_database():
    data=datasets.load_sales_data_cartoon_database()
    print(data)
    sales_table=data.sales_table
    print(sales_table)

def test_df2SQLite():
    data=datasets.load_sales_data_cartoon_database()
    sales_table=data.sales_table    
    db_fp="./database/fruits.sqlite"
    database.df2SQLite(db_fp,sales_table,"sales")  
    print(help(database.df2SQLite))
    
def test_SQLite2df():
    db_fp="./database/fruits.sqlite"
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
        
def test_coordinate_transformation():
    lng = 128.543
    lat = 37.065
    # datasets._baiduPOI_dataCrawler.wgs84togcj02(lng, lat)
    # print(help(bdc))
    from usda.utils import _coordinate_transformation as cc
    print(help(cc))
    
def test_baiduPOI_dataCrawler_circle():
    page_num_range=range(20)
    query_dic={
        'location':'34.265708,108.953431',
        'radius':1000,
        'query':'旅游景点',   
        'page_size':'20',
        'scope':2, 
        'output':'json',
        'ak':'YuN8HxzYhGNfNLGX0FVo3NU3NOrgSNdF'        
    }    
    poi_save_path='./data/poi_circle.json'    
    datasets.baiduPOI_dataCrawler_circle(query_dic,poi_save_path,page_num_range)
    
def test_csv2df():
    poi_fn_csv='./data/poi_csv.csv'
    data=database.csv2df(poi_fn_csv)
    print(data)
    
def test_filePath_extraction():
    dirpath='./test'
    fileType=["csv","sqlite"]
    poi_paths=utils.filePath_extraction(dirpath,fileType)
    print(poi_paths) 
    
def test_poi_csv2GeoDF_batch():
    pass

def test_ployly_table():
    poi_gdf=pd.read_pickle('./data/poisInAll_gdf.pkl')
    df=poi_gdf.loc[pd.IndexSlice[:,:2],:]
    df=df.reset_index()
    column_extraction=['level_0','name', 'location_lat', 'location_lng', 'detail_info_tag','detail_info_overall_rating', 'detail_info_price']
    charts.plotly_table(df,column_extraction) 

def test_frequency_bins():
    data=datasets.load_ramen_price_cartoon_statistic()
    # print(data)
    df=data.ramen_price
    bins=range(500,1000+100,100)  # 配置分割区间（组距）
    result=stats.frequency_bins(df,bins,"price")
    print(result)
    
def test_comparisonOFdistribution():
    poi_gdf=pd.read_pickle('./data/poisInAll_gdf.pkl')
    # print(poi_gdf.columns)
    delicacy_price=poi_gdf.xs('poi_0_delicacy',level=0).detail_info_price  # 提取美食价格数据
    delicacy_price_df=delicacy_price.to_frame(name='price').astype(float)
    delicacy_price_df_clean=delicacy_price_df.dropna()
    stats.comparisonOFdistribution(delicacy_price_df_clean,'price',bins=100)
        
def test_is_outlier():
    import numpy as np
    outlier_data=np.array([2.1,2.6,2.4,2.5,2.3,2.1,2.3,2.6,8.2,8.3])
    is_outlier_bool,data_clean=stats.is_outlier(outlier_data,threshold=3.5) 
    print(is_outlier_bool,data_clean)
    
def test_probability_graph():    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,10))
    charts.probability_graph(x_i=113,x_min=50,x_max=150,step=1,subplot_num=221,loc=100,scale=12)
    charts.probability_graph(x_i=113,x_min=50,x_max=150,step=1,left=False,subplot_num=223,loc=100,scale=12)
    charts.probability_graph(x_i=113,x_min=50,x_max=150,x_s=90,step=1,subplot_num=222,loc=100,scale=12)
    charts.probability_graph(x_i=90,x_min=50,x_max=150,step=1,subplot_num=224,loc=100,scale=20)
    plt.show()        

if __name__=="__main__":    
    print(dir(usda))
    print("-"*50)
    test_load_sales_data_cartoon_database()
    test_df2SQLite()
    test_SQLite2df()
    test_postSQL2gpd()
    test_DisplayablePath()    
    test_coordinate_transformation()
    # test_baiduPOI_dataCrawler_circle()
    test_csv2df()
    test_filePath_extraction()
    test_ployly_table()
    test_frequency_bins()
    test_comparisonOFdistribution()
    test_is_outlier()
    test_probability_graph()
```