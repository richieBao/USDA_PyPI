# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 10:44:32 2022

@author: richie bao 
"""
import usda
from usda import *
import pandas as pd

def test_shpPolygon2OsmosisTxt():
    shape_polygon_small_fp='./test/data/OSMBoundary_small/OSMBoundary_small.shp'
    osmosis_txt_small_fp='./test/data/OSMBoundary_small.txt'
    data_process.shpPolygon2OsmosisTxt(shape_polygon_small_fp,osmosis_txt_small_fp)
    
def test_osmHandler():
    osm_Chicago_fp=r"./test\data\osm_small_clip.osm" 
    osm_handler=data_process.osmHandler()
    osm_handler.apply_file(osm_Chicago_fp,locations=True) 
    print(osm_handler)
    
def test_save_osm():
    osm_Chicago_fp=r"./test\data\osm_small_clip.osm" 
    osm_handler=data_process.osmHandler()
    osm_handler.apply_file(osm_Chicago_fp,locations=True) 
    node_gdf=data_process.save_osm(osm_handler,osm_type="area",save_path="./test/data",fileType="GPKG")
    print(node_gdf)
    
def test_duration():
    class A(object): 
        def __init__(self,x):
            self.x=x
        
    start_time=utils.start_time()
    f=[A(523825) for i in range(1000000)]
    utils.duration(start_time)
    
def test_pts2raster():
    amenity_kde_fn='./test/data/amenity_kde/amenity_kde.shp'
    raster_path='./test/data/amenity_epsg32616.tif'
    cellSize=300
    field_name='amenityKDE'
    poiRaster_array=data_process.pts2raster(amenity_kde_fn,raster_path,cellSize,field_name)
    print("conversion completed!")    
    print(poiRaster_array)
    
def test_ptsKDE_geoDF2raster():
    import geopandas as gpd
    
    fn=r"./test/data/amenity_kde/amenity_kde.shp" 
    amenity_poi=gpd.read_file(fn)
    raster_path_gpd='./test/data/amenity_kde_gdf.tif'
    cellSize=500 # cellSize值越小，需要计算的时间越长，开始调试时，可以尝试将其调大以节约计算时间
    scale=10**10 # 相当于math.pow(10,10)
    poiRasterGeoDF_array=stats.ptsKDE_geoDF2raster(amenity_poi,raster_path_gpd,cellSize,scale)  
    print(poiRasterGeoDF_array)
        
def test_print_html():
    dataFp_dic={
    "ublic_Health_Statistics_byCommunityArea_fp":r'./test/data/Public_Health_Statistics_Selected_public_health_indicators_by_Chicago_community_area.csv',
    "Boundaries_Community_Areas_current":r'./test/data/ChicagoCommunityAreas/ChicagoCommunityAreas.shp',    
    }
    
    pubicHealth_Statistic=pd.read_csv(dataFp_dic["ublic_Health_Statistics_byCommunityArea_fp"])
    html=charts.print_html(pubicHealth_Statistic,6)
    print(html)

def test_generate_colors():
    a,b,c=charts.generate_colors()
    print(b)
    
def test_demo_con_style():
    from sympy import Symbol
    import sympy
    import matplotlib.pyplot as plt
    
    emma_statureAge={"age":list(range(4,20)),"stature":[100.1,107.2,114.1,121.7,126.8,130.9,137.5,143.2,149.4,151.6,154.0,154.6,155.0,155.1,155.3,155.7]}
    emma_statureAge_df=pd.DataFrame(emma_statureAge)
    fig, ax=plt.subplots(figsize=(8,8))
    x=Symbol('x')
    f_emma=-326.6/x+173.3
    f_emma_=sympy.lambdify(x,f_emma,"numpy")
    ax.plot(emma_statureAge_df.age,f_emma_(emma_statureAge_df.age),'o-',label='$-326.6/x+173.3$')
    
    ax.plot(emma_statureAge_df.age,emma_statureAge_df.stature,'o',label='ground truth',color='r')
    ax.plot(emma_statureAge_df.age,f_emma_(emma_statureAge_df.age),'o-',label='$-326.6/x+173.3$')
            
    dx=3
    charts.demo_con_style((6,f_emma.evalf(subs={x:6})),(6+dx,f_emma.evalf(subs={x:6+dx})),ax,"angle,angleA=-90,angleB=180,rad=0")    
    ax.text(7, f_emma.evalf(subs={x:6})-3, "△ x", family="monospace",size=20)
    ax.text(9.3, f_emma.evalf(subs={x:9.3})-10, "△ y", family="monospace",size=20)
    
def test_coefficient_of_determination():
    import sympy
    
    dt=pd.date_range('2020-07-22', periods=14, freq='D')
    dt_temperature_iceTeaSales={"dt":dt,"temperature":[29,28,34,31,25,29,32,31,24,33,25,31,26,30],"iceTeaSales":[77,62,93,84,59,64,80,75,58,91,51,73,65,84]}
    iceTea_df=pd.DataFrame(dt_temperature_iceTeaSales).set_index("dt")
    
    # B - 使用sklearn库sklearn.linear_model.LinearRegression()，Ordinary least squares Linear Regression-普通最小二乘线性回归，获取回归方程
    from sklearn.linear_model import LinearRegression
    X,y=iceTea_df.temperature.to_numpy().reshape(-1,1),iceTea_df.iceTeaSales.to_numpy()
    
    # 拟合模型
    LR=LinearRegression().fit(X,y)
    y_pre=LR.predict(X)
    R_square_a,R_square_b=stats.coefficient_of_determination(iceTea_df.iceTeaSales.to_list(),y_pre)  
    print(R_square_a,R_square_b)
            
def test_ANOVA():
    import sympy
    
    dt=pd.date_range('2020-07-22', periods=14, freq='D')
    dt_temperature_iceTeaSales={"dt":dt,"temperature":[29,28,34,31,25,29,32,31,24,33,25,31,26,30],"iceTeaSales":[77,62,93,84,59,64,80,75,58,91,51,73,65,84]}
    iceTea_df=pd.DataFrame(dt_temperature_iceTeaSales).set_index("dt")
    
    # B - 使用sklearn库sklearn.linear_model.LinearRegression()，Ordinary least squares Linear Regression-普通最小二乘线性回归，获取回归方程
    from sklearn.linear_model import LinearRegression
    X,y=iceTea_df.temperature.to_numpy().reshape(-1,1),iceTea_df.iceTeaSales.to_numpy()
    LR=LinearRegression().fit(X,y)
    y_pre=LR.predict(X)
    stats.ANOVA(iceTea_df.iceTeaSales.to_list(),y_pre,df_reg=1,df_res=12) 
    
def test_confidenceInterval_estimator_LR():
    dt=pd.date_range('2020-07-22', periods=14, freq='D')
    dt_temperature_iceTeaSales={"dt":dt,"temperature":[29,28,34,31,25,29,32,31,24,33,25,31,26,30],"iceTeaSales":[77,62,93,84,59,64,80,75,58,91,51,73,65,84]}
    iceTea_df=pd.DataFrame(dt_temperature_iceTeaSales).set_index("dt")
    from sklearn.linear_model import LinearRegression
    X,y=iceTea_df.temperature.to_numpy().reshape(-1,1),iceTea_df.iceTeaSales.to_numpy()
    LR=LinearRegression().fit(X,y)    
    
    sample_num=14
    confidence=0.05
    iceTea_df_sort=iceTea_df.sort_values(by=['temperature'])
    X,y=iceTea_df_sort.temperature.to_numpy().reshape(-1,1),iceTea_df_sort.iceTeaSales.to_numpy()
    CI=stats.confidenceInterval_estimator_LR(27,sample_num,X,y,LR,confidence)       
     
def test_correlationAnalysis_multivarialbe():
    import pandas as pd
    
    store_info={"location":['Ill.','Ky.','Lowa.','Wis.','MIch.','Neb.','Ark.','R.I.','N.H.','N.J.'],"area":[10,8,8,5,7,8,7,9,6,9],"distance_to_nearestStation":[80,0,200,200,300,230,40,0,330,180],"monthly_turnover":[469,366,371,208,246,297,363,436,198,364]}
    storeInfo_df=pd.DataFrame(store_info)
    
    p_values,correlation=stats.correlationAnalysis_multivarialbe(storeInfo_df)
    print("p_values:")
    print(p_values)
    print("_"*78)
    print("correlation:")
    print(correlation)    
    

if __name__=="__main__":
    print(dir(usda))    
    print("-"*50)
    # test_shpPolygon2OsmosisTxt()
    # test_osmHandler()
    # test_save_osm()
    # test_duration()
    # test_pts2raster()
    # test_ptsKDE_geoDF2raster()
    # test_print_html()
    # test_generate_colors()
    # test_demo_con_style()
    # test_coefficient_of_determination()
    # test_ANOVA()
    # test_confidenceInterval_estimator_LR()
    # test_correlationAnalysis_multivarialbe()
    
    