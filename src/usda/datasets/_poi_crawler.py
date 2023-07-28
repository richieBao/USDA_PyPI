# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 18:46:09 2023

@author: richie bao
"""
if __package__:
    from ._coordinate_transformation import bd09togcj02
    from ._coordinate_transformation import gcj02towgs84
else:
    from _coordinate_transformation import bd09togcj02
    from _coordinate_transformation import gcj02towgs84

import urllib,json,csv,os,pathlib
from tqdm import tqdm

def baiduPOI_dataCrawler(query_dic,bound_coordinate,partition,page_num_range,poi_fn_list=False):
    '''
    function - 百度地图开放平台POI数据检索——多边形区域检索（矩形区域检索）方式
    
    Params:
        query_dic - 请求参数配置字典，详细参考上文或者百度服务文档；dict
        bound_coordinate - 以字典形式配置下载区域；dict
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
            else:
                poi_csv=None
            if fP.suffix=='.json':
                poi_json=open(fP,'w',encoding='utf-8')
            else:
                poi_json=None
    num=0
    jsonDS=[]  # 存储读取的数据，用于.json格式数据的保存
    # 循环切分的检索区域，逐区下载数据
    print("Start downloading data...")
    for i in tqdm(range(partition)):
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
                # print(query_dic)
                url=urlRoot+urllib.parse.urlencode(query_dic)
                data=urllib.request.urlopen(url)
                responseOfLoad=json.loads(data.read())     
                print(url,responseOfLoad.get("message"))
                if responseOfLoad.get("message")=='ok':
                    results=responseOfLoad.get("results") 
                    for row in range(len(results)):
                        subData=results[row]
                        baidu_coordinateSystem=[subData.get('location').get('lng'),subData.get('location').get('lat')]  # 获取百度坐标系
                        Mars_coordinateSystem=bd09togcj02(baidu_coordinateSystem[0], baidu_coordinateSystem[1])  # 百度坐标系-->火星坐标系
                        WGS84_coordinateSystem=gcj02towgs84(Mars_coordinateSystem[0], Mars_coordinateSystem[1])  # 火星坐标系-->WGS84
                        
                        # 更新坐标
                        subData['location']['lat']=WGS84_coordinateSystem[1]
                        subData['detail_info']['lat']=WGS84_coordinateSystem[1]
                        subData['location']['lng']=WGS84_coordinateSystem[0]
                        subData['detail_info']['lng']=WGS84_coordinateSystem[0]  
                        # print('+++')
                        # print(subData)
                        if csv_writer:
                            csv_writer.writerow([subData])  # 逐行写入.csv文件
                        jsonDS.append(subData)
            num+=1       
            # print("No."+str(num)+" was written to the .csv file.")
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
        query_dic - 请求参数配置字典，详细参考上文或者百度服务文档；dict
        poi_save_path - 存储文件路径；string
        page_num_range - 配置页数范围；range()
        
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
                Mars_coordinateSystem=bd09togcj02(baidu_coordinateSystem[0], baidu_coordinateSystem[1])  # 百度坐标系-->火星坐标系
                WGS84_coordinateSystem=gcj02towgs84(Mars_coordinateSystem[0],Mars_coordinateSystem[1])  # 火星坐标系-->WGS84

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

if __name__=="__main__":      
    poi_classification=['美食','酒店','购物','生活服务','丽人','旅游景点','休闲娱乐','运动健身','教育培训','文化传媒','医疗','汽车服务','交通设施','金融','房地产','公司企业','政府机构']
    
    # bound_coordinate={'leftBottom':[108.921815,34.258596],'rightTop':[108.986888,34.273334]} 
    # page_num_range=range(20)
    # partition=4 
    
    # data_path=r'F:\data\POI_dongxistreet' 
    # for classi in poi_classification:
    #     poi_fn_csv=os.path.join(data_path,f'poi_{classi}.csv')
    #     query_dic={
    #         'query':classi,
    #         'page_size':'20', 
    #         'scope':2,
    #         'ak':'jKT18BmG1LKYcICQbCu432lnbCkaPG8p', 
    #     }       
    #     baiduPOI_dataCrawler(query_dic,bound_coordinate,partition,page_num_range,poi_fn_list=[poi_fn_csv])  
        
    #     break    
    
    #--------------------------------------------------------------------------
    
    page_num_range=range(20)
    data_path=r'F:\data\POI_dongxistreet\circle_search_3000m' 
    for classi in poi_classification:
        poi_save_path=os.path.join(data_path,f'poi_{classi}.json')    

        query_dic={
            'location':'34.265757,108.953489',  # 108.953489,34.265757
            'radius':500,
            'query':classi,   
            'page_size':'20',
            'scope':2, 
            'output':'json',
            'ak':'jKT18BmG1LKYcICQbCu432lnbCkaPG8p'        
        }  

        baiduPOI_dataCrawler_circle(query_dic,poi_save_path,page_num_range)
        
        break
        
        

