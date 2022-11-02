# usda.datasets

加载数据集。

---

?> __测试用假数据__

## datasets.load_sales_data_cartoon_database

测试用数据。根据《漫画数据库》一书中的销售数据集构建

```python
def load_sales_data_cartoon_database(data_module=DATA_MODULE):
    测试用数据。根据《漫画数据库》一书中的销售数据集构建

    Parameters
    ----------
    data_module : TYPE, string
        数据所在文件夹. The default is DATA_MODULE.

    Returns
    -------
    Type, Class
        假数据：销售数据.含属性（字段）：'sales_table','exporting_country_table','sale_details_table','commodity_table'
```

## datasets.load_ramen_price_cartoon_statistic

源于《漫画统计学》中“美味拉面畅销前50”上刊载的拉面馆的拉面价格

```python
load_ramen_price_cartoon_statistic(data_module='usda.datasets.data')
    源于《漫画统计学》中“美味拉面畅销前50”上刊载的拉面馆的拉面价格    
    
    Parameters
    ----------
    data_module : string, optional
        数据所在文件夹. The default is DATA_MODULE.
    
    Returns
    -------
    Class
        拉面假数据：含属性字段ramen_price，file_name.
```

## datasets.load_bowling_contest_cartoon_statistic

数据源于《漫画统计学》保龄球大赛的结果

```python
load_bowling_contest_cartoon_statistic(data_module='usda.datasets.data')
    数据源于《漫画统计学》保龄球大赛的结果
    
    Parameters
    ----------
    data_module : string, optional
        数据所在文件夹. The default is DATA_MODULE.
    
    Returns
    -------
    Class
        保龄球大赛得分，含属性字段bowling_contest，file_name.
```

## datasets.load_test_score_cartoon_statistic

源于《漫画统计学》中的考试成绩数据

```python
load_test_score_cartoon_statistic(data_module='usda.datasets.data')
    源于《漫画统计学》中的考试成绩数据
    
    Parameters
    ----------
    data_module : string, optional
        数据所在文件夹. The default is DATA_MODULE.
    
    Returns
    -------
    Class
        试成绩数据，含属性字段test_score，file_name.
```

?> 数据检索

## datasets.baiduPOI_dataCrawler

百度地图开放平台POI数据检索——多边形区域检索（矩形区域检索）方式。索目前为高级权限，如有需求，需要在百度地图开放平台上提交工单咨询

```python
baiduPOI_dataCrawler(query_dic, bound_coordinate, partition, page_num_range, poi_fn_list=False)
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
```

## datasets.baiduPOI_dataCrawler_circle

百度地图开放平台POI数据检索——圆形区域检索方式

```python 
baiduPOI_dataCrawler_circle(query_dic, poi_save_path, page_num_range)
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
```

## datasets.baiduPOI_batchCrawler

百度地图开放平台POI数据批量爬取，需要调用单个分类POI检索函数

```python
baiduPOI_batchCrawler(poi_config_para)
    function - 百度地图开放平台POI数据批量爬取，
               需要调用单个分类POI检索函数 

    baiduPOI_dataCrawler(query_dic,bound_coordinate,partition,page_num_range,poi_fn_list=False)
    
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
```

?> KITTI数据集相关信息提取

## datasets.KITTI_info

读取KITTI文件信息，1-包括经纬度，惯性导航系统信息等的.txt文件，2-包含时间戳的.txt文件

```python
KITTI_info(KITTI_info_fp, timestamps_fp)
    function - 读取KITTI文件信息，1-包括经纬度，惯性导航系统信息等的.txt文件，2-包含时间戳的.txt文件
    
    Params:
        KITTI_info_fp - 数据文件路径；string
        timestamps_fp - 时间戳文件路径；string
        
    Returns:
        drive_info - 返回数据；DataFrame
```

## datasets.KITTI_info_gap

读取KITTI文件信息，1-包括经纬度，惯性导航系统信息等的.txt文件。只返回经纬度、海拔信息

```python
KITTI_info_gap(KITTI_info_fp, save_fp, gap=1)
    function - 读取KITTI文件信息，1-包括经纬度，惯性导航系统信息等的.txt文件。只返回经纬度、海拔信息
    
    Params:
        KITTI_info_fp - 数据文件路径；string
        save_fp - 文件保存路径；string
        gap - 间隔连续剔除部分图像避免干扰， 默认值为1；int
        
    Returns:
        drive_info_coordi - 返回经纬度和海拔信息；DataFrame
```

?> Sentinel-2 遥感影像信息

## datasets.Sentinel2_bandFNs

获取sentinel-2波段文件路径，和打印主要信息

```python
Sentinel2_bandFNs(MTD_MSIL2A_fn)
    funciton - 获取sentinel-2波段文件路径，和打印主要信息
    
    Params:
        MTD_MSIL2A_fn - MTD_MSIL2A 文件路径；string
    
    Returns:
        band_fns_list - 波段相对路径列表；list(string)
        band_fns_dict - 波段路径为值，反应波段信息的字段为键的字典；dict
```

---

## datasets.kml_coordiExtraction

提取.kml文件中的坐标信息

```python
kml_coordiExtraction(kml_pathDict)
    function - 提取.kml文件中的坐标信息
    
    Params:
        kml_pathDict - .kml文件路径字典。文件夹名为键，值为包含该文件夹下所有文件名的列表。使用filePath_extraction()函数提取。
    
    Returns:
        kml_coordi_dict - 返回坐标信息；dict
```

## datasets.img_exif_info

提取数码照片的属性信息和拍摄数据，即可交换图像文件格式（Exchangeable image file format，Exif）

```python
img_exif_info(img_fp, printing=True)
    function - 提取数码照片的属性信息和拍摄数据，即可交换图像文件格式（Exchangeable image file format，Exif）
    
    Params:
        img_fp - 一个图像的文件路径；string
        printing - 是否打印。The default is True；bool
        
    Returns:
        exif_ - 提取的照片信息结果；dict
```