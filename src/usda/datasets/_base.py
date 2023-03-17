# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 19:58:26 2022

@author: richie bao
"""
import pickle
import os
from importlib import resources
from ..utils import Bunch
import urllib
import json
import csv
import pathlib
from ..utils import _coordinate_transformation as cc
from tqdm import tqdm
import numpy as np
from scipy import linalg, ndimage
import pandas as pd

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
                 exporting_country_table=exporting_country_table,
                 sale_details_table=sale_details_table,
                 commodity_table=commodity_table,
                 file_name=data_file_name)

def load_ramen_price_cartoon_statistic(data_module=DATA_MODULE):
    '''
    源于《漫画统计学》中“美味拉面畅销前50”上刊载的拉面馆的拉面价格    

    Parameters
    ----------
    data_module : string, optional
        数据所在文件夹. The default is DATA_MODULE.

    Returns
    -------
    Class
        拉面假数据：含属性字段ramen_price，file_name.

    '''
    data_file_name="ramen_price_cartoon_statistic.pickle" 
    with resources.open_binary(data_module,data_file_name) as data_file:
        data=pickle.load(data_file)  
        
        data_file_name="ramen_price_cartoon_statistic"
    
    return Bunch(ramen_price=data, file_name=data_file_name)

def load_bowling_contest_cartoon_statistic(data_module=DATA_MODULE):
    '''
    数据源于《漫画统计学》保龄球大赛的结果

    Parameters
    ----------
    data_module : string, optional
        数据所在文件夹. The default is DATA_MODULE.

    Returns
    -------
    Class
        保龄球大赛得分，含属性字段bowling_contest，file_name.

    '''
    data_file_name="bowling_contest_cartoon_statistic.pickle" 
    with resources.open_binary(data_module,data_file_name) as data_file:
        data=pickle.load(data_file)  
        
        data_file_name="bowling_contest_cartoon_statistic"
    
    return Bunch(bowling_contest=data, file_name=data_file_name)

def load_test_score_cartoon_statistic(data_module=DATA_MODULE):
    '''
    源于《漫画统计学》中的考试成绩数据

    Parameters
    ----------
    data_module : string, optional
        数据所在文件夹. The default is DATA_MODULE.

    Returns
    -------
    Class
        试成绩数据，含属性字段test_score，file_name.

    '''
    data_file_name="test_score_cartoon_statistic.pickle" 
    with resources.open_binary(data_module,data_file_name) as data_file:
        data=pickle.load(data_file)  
        
        data_file_name="test_score_cartoon_statistic"
    
    return Bunch(test_score=data, file_name=data_file_name)

def baiduPOI_dataCrawler(query_dic,bound_coordinate,partition,page_num_range,poi_fn_list=False):
    '''
    function - 百度地图开放平台POI数据检索——多边形区域检索（矩形区域检索）方式。
    多边形区域检索目前为高级权限，如有需求，需要在百度地图开放平台上提交工单咨询。    
    
    Params:
        query_dic - 请求参数配置字典，详细参考上文或者百度服务文档；例如：query_dic={'query':'旅游景点','page_size':'20','scope':2, 'ak':从百度地图开放平台申请' }
        bound_coordinate - 以字典形式配置下载区域；，例如：{'leftBottom':[108.776852,34.186027],'rightTop':[109.129275,34.382171]}
        partition - 检索区域切分次数；int
        page_num_range - 配置页数范围；range()
        poi_fn_list=False - 定义的存储文件名列表；list
        
    Returns:
        None
    '''   
    urlRoot='http://api.map.baidu.com/place/v2/search?'  # 数据下载网址，查询百度地图服务文档
    # 切分检索区域
    if bound_coordinate:
        xDis=(bound_coordinate['rightTop'][0]-bound_coordinate['leftBottom'][0])/partition
        yDis=(bound_coordinate['rightTop'][1]-bound_coordinate['leftBottom'][1])/partition    
    # 判断是否要写入文件
    if poi_fn_list:
        for file_path in poi_fn_list:
            fP=pathlib.Path(file_path)
            if fP.suffix=='.csv':
                poi_csv=open(fP,'w',encoding='utf-8')
                csv_writer=csv.writer(poi_csv)    
            elif fP.suffix=='.json':
                poi_json=open(fP,'w',encoding='utf-8')
    num=0
    jsonDS=[]  # 存储读取的数据，用于.json格式数据的保存
    # 循环切分的检索区域，逐区下载数据
    print("Start downloading data...")
    for i in range(partition):
        for j in range(partition):
            leftBottomCoordi=[bound_coordinate['leftBottom'][0]+i*xDis,bound_coordinate['leftBottom'][1]+j*yDis]
            rightTopCoordi=[bound_coordinate['leftBottom'][0]+(i+1)*xDis,bound_coordinate['leftBottom'][1]+(j+1)*yDis]
            for p in page_num_range:  
                # 更新请求参数
                query_dic.update({'page_num':str(p),
                                  'bounds':str(leftBottomCoordi[1]) + ',' + str(leftBottomCoordi[0]) + ',' + 
                                           str(rightTopCoordi[1]) + ',' + str(rightTopCoordi[0]),
                                  'output':'json',
                                 })
                
                url=urlRoot+urllib.parse.urlencode(query_dic)
                data=urllib.request.urlopen(url)
                responseOfLoad=json.loads(data.read())     
                # print(url,responseOfLoad.get("message"))
                if responseOfLoad.get("message")=='ok':
                    results=responseOfLoad.get("results") 
                    for row in range(len(results)):
                        subData=results[row]
                        baidu_coordinateSystem=[subData.get('location').get('lng'),subData.get('location').get('lat')]  # 获取百度坐标系
                        Mars_coordinateSystem=cc.bd09togcj02(baidu_coordinateSystem[0], baidu_coordinateSystem[1])  # 百度坐标系-->火星坐标系
                        WGS84_coordinateSystem=cc.gcj02towgs84(Mars_coordinateSystem[0], Mars_coordinateSystem[1])  # 火星坐标系-->WGS84
                        
                        # 更新坐标
                        subData['location']['lat']=WGS84_coordinateSystem[1]
                        subData['detail_info']['lat']=WGS84_coordinateSystem[1]
                        subData['location']['lng']=WGS84_coordinateSystem[0]
                        subData['detail_info']['lng']=WGS84_coordinateSystem[0]   
                        if csv_writer:
                            csv_writer.writerow([subData])  # 逐行写入.csv文件
                        jsonDS.append(subData)
            num+=1       
            print("No."+str(num)+" was written to the .csv file.")
    if poi_json:       
        json.dump(jsonDS,poi_json)
        poi_json.write('\n')
        poi_json.close()
    if poi_csv:
        poi_csv.close()
    print("The download is complete.")

def baiduPOI_dataCrawler_circle(query_dic,poi_save_path,page_num_range):
    '''
    function - 百度地图开放平台POI数据检索——圆形区域检索方式
    
    Params:
        query_dic - 请求参数配置字典，详细参考上文或者百度服务文档；dict，例如：
                    query_dic={
                            'location':'34.265708,108.953431',
                            'radius':1000,
                            'query':'旅游景点',   
                            'page_size':'20',
                            'scope':2, 
                            'output':'json',
                            'ak':'YuN8HxzYhGNfNLGX0FVo3NU3NOrgSNdF'        
                        } 
        poi_save_path - 存储文件路径；string
        page_num_range - 配置页数范围；range()，例如range(20)
        
    Returns:
        None
    '''   
    urlRoot='http://api.map.baidu.com/place/v2/search?' #数据下载网址，查询百度地图服务文档
    poi_json=open(poi_save_path,'w',encoding='utf-8')  
    jsonDS=[]  # 存储读取的数据，用于.json格式数据的保存
    for p in tqdm(page_num_range): 
        # 更新请求参数
        query_dic.update({'page_num':str(p)})
        url=urlRoot+urllib.parse.urlencode(query_dic)
        data=urllib.request.urlopen(url)
        responseOfLoad=json.loads(data.read())     
        if responseOfLoad.get("message")=='ok':
            results=responseOfLoad.get("results") 
            for row in range(len(results)):
                subData=results[row]
                baidu_coordinateSystem=[subData.get('location').get('lng'),subData.get('location').get('lat')]  # 获取百度坐标系
                Mars_coordinateSystem=cc.bd09togcj02(baidu_coordinateSystem[0], baidu_coordinateSystem[1])  # 百度坐标系-->火星坐标系
                WGS84_coordinateSystem=cc.gcj02towgs84(Mars_coordinateSystem[0],Mars_coordinateSystem[1])  # 火星坐标系-->WGS84

                # 更新坐标
                subData['location']['lat']=WGS84_coordinateSystem[1]
                subData['detail_info']['lat']=WGS84_coordinateSystem[1]
                subData['location']['lng']=WGS84_coordinateSystem[0]
                subData['detail_info']['lng']=WGS84_coordinateSystem[0]  
                jsonDS.append(subData)

    if poi_json:       
        json.dump(jsonDS,poi_json)
        poi_json.write('\n')
        poi_json.close()
    print("The download is complete.")       

def baiduPOI_batchCrawler(poi_config_para):    
    '''
    function - 百度地图开放平台POI数据批量爬取，
               需要调用单个分类POI检索函数 baiduPOI_dataCrawler(query_dic,bound_coordinate,partition,page_num_range,poi_fn_list=False)
    
    Paras:
        poi_config_para - 参数配置，包含：
            'data_path'（配置数据存储位置），
            'bound_coordinate'（矩形区域检索坐下、右上经纬度坐标），
            'page_num_range'（配置页数范围），
            'partition'（检索区域切分次数），
            'page_size'（单次召回POI数量），
            'scope'（检索结果详细程度），
            'ak'（开发者的访问密钥）
            
    Returns:
        None
    ''' 
    for idx,(poi_ClassiName,poi_classMapping) in enumerate(poi_classificationName.items()):
        print(str(idx+16)+"_"+poi_ClassiName)
        poi_subFileName="poi_"+str(idx+16)+"_"+poi_classMapping
        data_path=poi_config_para['data_path']
        poi_fn_csv=os.path.join(data_path,poi_subFileName+'.csv')
        poi_fn_json=os.path.join(data_path,poi_subFileName+'.json')
        
        query_dic={
            'query':poi_ClassiName,
            'page_size':poi_config_para['page_size'],
            'scope':poi_config_para['scope'],
            'ak':poi_config_para['ak']                        
        }
        bound_coordinate=poi_config_para['bound_coordinate']
        partition=poi_config_para['partition']
        page_num_range=poi_config_para['page_num_range']
        util_A.baiduPOI_dataCrawler(query_dic,bound_coordinate,partition,page_num_range,poi_fn_list=[poi_fn_csv,poi_fn_json]) 
        

def generate_categorical_2darray(**kwargs):    
    '''
    生成具有一定聚类二维空间分布特征的分类属性矩阵数据
    ref:Feature agglomeration vs. univariate selection, https://scikit-learn.org/stable/auto_examples/cluster/plot_feature_agglomeration_vs_univariate_selection.html#sphx-glr-auto-examples-cluster-plot-feature-agglomeration-vs-univariate-selection-py

    Parameters
    ----------
    **kwargs : kwargs
        默认值为：
        args=dict(
            size=16,二维矩阵大小；numerical
            n_samples=3, 样本数量；int
            seed=None,随机种子；int
            snr=5.0,噪声系数；numerical
            )  .

    Returns
    -------
    X : ndarray
        样本数组.
    y : 1darray
        类标.

    '''
    
    args=dict(
        size=16,
        n_samples=3,
        seed=None,
        snr=5.0,
        sigma=1.0,
        )    
    
    args['roi_size']=args['size']-1
    args.update(kwargs)
    
    roi_size=args['roi_size']
    size=args['size']
    
    if args['seed']:
        np.random.seed(args['seed'])
    else:
        np.random.seed()
        
    coef=np.zeros((size, size))
    coef[0:roi_size, 0:roi_size]=-1.0
    coef[-roi_size:, -roi_size:]=1.0
    
    X=np.random.randn(args['n_samples'], size**2)
    for x in X:  # smooth data
        x[:]=ndimage.gaussian_filter(x.reshape(size, size), sigma=args['sigma']).ravel()
    X -= X.mean(axis=0)    
    X /= X.std(axis=0)
    y=np.dot(X, coef.ravel())
    
    noise=np.random.randn(y.shape[0])
    noise_coef=(linalg.norm(y, 2) / np.exp(args['snr'] / 20.0)) / linalg.norm(noise, 2)
    y += noise_coef * noise
    
    return X,y        

def load_evaluation_criteria_raw_values(data_module=DATA_MODULE):
    '''
    来自于：Boroushaki, S. Entropy-Based Weights for MultiCriteria Spatial Decision-Making. Yearbook of the Association of Pacific Coast Geographers 79, 168–187 (2017).一文中的演示数据
    用于说明信息熵权重

    Parameters
    ----------
    data_module : string
        数据所在文件夹. The default is DATA_MODULE.

    Returns
    -------
    Class
        含属性字段：DataFrame数据及文件名.

    '''
    
    data_file_name="evaluation_criteria_raw_values.pickle" 
    with resources.open_binary(data_module,data_file_name) as data_file:
        evaluation_criteria_raw_values=pd.read_pickle(data_file)  
    return Bunch(data=evaluation_criteria_raw_values, name=data_file_name)

def load_sustainability_attributes4electricity_generation_tech(data_module=DATA_MODULE):
    '''
    来自于： Şahin, M. A comprehensive analysis of weighting and multicriteria methods in the context of sustainable energy. International Journal of Environmental Science and Technology 18, 1591–1616 (2021).

    Parameters
    ----------
    data_module : string
        数据所在文件夹. The default is DATA_MODULE.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    
    data_file_name="sustainability_attributes4electricity_generation_tech.pickle" 
    with resources.open_binary(data_module,data_file_name) as data_file:
        data=pd.read_pickle(data_file)  
    return Bunch(data=data, name=data_file_name) 

def load_microclimate_in_office_rooms(data_module=DATA_MODULE):
    data_file_name="microclimate_in_office_rooms.pickle" 
    with resources.open_binary(data_module,data_file_name) as data_file:
        data=pd.read_pickle(data_file)  
    measurement_units=[r'm3/h', r'%', r'°C', r'lx', r'm/s', r'°C']
    optimisation_direction=[r'max',r'max',r'max',r'max',r'min',r'min']
    weight_of_criteria=[0.21, 0.16, 0.26, 0.17, 0.12, 0.08] 
    optimal_value=[15, 50, 24.5, 400, 0.05, 5] 
    
    return Bunch(data=data, 
                 name=data_file_name,
                 measurement_units=measurement_units,
                 optimisation_direction=optimisation_direction,
                 weight_of_criteria=weight_of_criteria,
                 optimal_value=optimal_value) 
    
def load_jisperveld_data(data_module=DATA_MODULE):
    '''
    来自于： Janssen, R., van Herwijnen, M., Stewart, T. J. & Aerts, J. C. J. H. Multiobjective decision support for land-use planning. Environ Plann B Plann Des 35, 740–756 (2008).

    Parameters
    ----------
    data_module : string
        数据所在文件夹. The default is DATA_MODULE.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''    
    
    data_file_name="jisperveld_data.pickle" 
    with resources.open_binary(data_module,data_file_name) as data_file:
        data=pd.read_pickle(data_file)     
        
    nature_recreation_vals={'nature_value':{'intensive_agriculture':4, 'extensive_agriculture':'nature_vals', 'residence':3,'industry':1,'recreation_day_trips':5,'recreation_overnight':5,'wet_natural_area':'nature_vals','water_recreational_use':7, 'water_limited_access':'nature_vals'},
                            'recreational_value':{'intensive_agriculture':6, 'extensive_agriculture':'nature_vals', 'residence':3,'industry':1,'recreation_day_trips':'recreation_b','recreation_overnight':'recreation_c','wet_natural_area':7,'water_recreational_use':'recreation_b', 'water_limited_access':1}}
    nature_recreation_vals_df=pd.DataFrame(nature_recreation_vals)     
    
    
    lu_conversion_cost=np.array([[0,-75,150,150,-225,0,-150,-300,-300],
          [75,0,150,150,-150,75,-75,-225,-225],
          [np.nan,np.nan,0,np.nan,-10000,-10000,np.nan,np.nan,np.nan],
          [np.nan,np.nan,np.nan,0,-10000,-10000,np.nan,np.nan,np.nan],
          [150,75,3,300,0,150,0,-150,-150],
          [0,-75,150,150,-150,0,-150,-300,-230],
          [np.nan,75,225,225,-75,150,0,-75,-75],
          [100,100,np.nan,np.nan,np.nan,np.nan,0,0,15],
          [100,100,np.nan,np.nan,np.nan,np.nan,0,0,0]])
    cols=['intensive_agriculture', 'extensive_agriculture','residence', 'industry','recreation_day_trips','recreation_overnight','wet_natural_area','water_recreational_use', 'water_limited_access']
    lu_conversion_cost_df=pd.DataFrame(lu_conversion_cost,index=cols,columns=cols)
            
    return Bunch(
        lu=data['lu'],
        nature_vals=data['nature_vals'],
        recreation_b=data['recreation_b'],
        recreation_c=data['recreation_c'],
        fixed_LU=data['fixed_LU'],
        lu_name={1:'intensive_agriculture',
                 2:'extensive_agriculture',
                 3:'residence',
                 4:'industry',
                 5:'recreation_day_trips',
                 6:'recreation_overnight',
                 7:'wet_natural_area',
                 8:'water_recreational_use',
                 9:'water_limited_access'},        
        nature_recreation_vals=nature_recreation_vals_df,
        lu_conversion_cost=lu_conversion_cost_df,
        data_name_lst=['lu','recreation_b','recreation_c','fixed_LU','nature_recreation_vals','lu_conversion_cost'],)
        

if __name__=="__main__":
    pass
    # sales_data_cartoon_databas=load_sales_data_cartoon_database()
    # df=ArithmeticErrorload_evaluation_criteria_raw_values
    # load_jisperveld_data()