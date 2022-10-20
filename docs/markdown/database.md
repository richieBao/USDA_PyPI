# usda.database

数据库，数据文件读写、编辑方法。

---

?> __SQLite数据库读写方法__

## database.df2SQLite

把pandas DataFrame格式数据写入数据库（同时创建表）。

```python
df2SQLite(db_fp, df, table_name, method='fail')
    function - 把pandas DataFrame格式数据写入数据库（同时创建表）
    
    Paras:
        db_fp - 数据库链接；string
        df - 待写入数据库的DataFrame格式数据；DataFrame
        table - 表名称；string
        method - 写入方法，'fail'，'replace'或'append'；string
    Returns:
        None
```

## database.SQLite2df

pandas方法，从SQLite数据库中读取表数据。

```python
SQLite2df(db_fp, table)
    function - pandas方法，从SQLite数据库中读取表数据
    
    Paras:
        db_fp - 数据库文件路径；string
        table - 所要读取的表；string
    
    Returns:
        读取的表；DataFrame
```



?> __PostgreSQL数据库读写方法__

## database.gpd2postSQL

将GeoDataFrame格式数据写入PostgreSQL数据库。

```python
gpd2postSQL(gdf, table_name, **kwargs)
    function - 将GeoDataFrame格式数据写入PostgreSQL数据库
    
    Paras:
        gdf - GeoDataFrame格式数据，含geometry字段（几何对象，点、线和面，数据值对应定义的坐标系统）；GeoDataFrame
        table_name - 写入数据库中的表名；string
        **kwargs - 连接数据库相关信息，包括myusername（数据库的用户名），mypassword（用户密钥），mydatabase（数据库名）；string
        
    Returns:
        None
```

## database.postSQL2gpd

读取PostgreSQL数据库中的表为GeoDataFrame格式数据。

```python
postSQL2gpd(table_name, geom_col='geometry', **kwargs)
    function - 读取PostgreSQL数据库中的表为GeoDataFrame格式数据
    
    Paras:
        table_name - 待读取数据库中的表名；string
        geom_col='geometry' - 几何对象，常规默认字段为'geometry'；string
        **kwargs - 连接数据库相关信息，包括myusername（数据库的用户名），mypassword（用户密钥），mydatabase（数据库名）；string
    Returns:
        读取的表数据；GeoDataFrame
```

?> __数据类型转换__

## database.csv2df

转换CSV格式的POI数据为pandas的DataFrame

```python
csv2df(poi_fn_csv)
    function-转换CSV格式的POI数据为pandas的DataFrame
    
    Params:
        poi_fn_csv - 存储有POI数据的CSV格式文件路径        
    
    Returns:
        poi_df - DataFrame(pandas)

None
```

## database.poi_csv2GeoDF_batch

CSV格式POI数据批量转换为GeoDataFrame格式数据，需要调用转换CSV格式的POI数据为pandas的DataFrame函数`csv2df(poi_fn_csv)`

```python
poi_csv2GeoDF_batch(poi_paths, fields_extraction, save_path)
    funciton - CSV格式POI数据批量转换为GeoDataFrame格式数据，需要调用转换CSV格式的POI数据为pandas的DataFrame函数csv2df(poi_fn_csv)
    
    Params:
        poi_paths - 文件夹路径为键，值为包含该文件夹下所有文件名列表的字典；dict
        fields_extraction - 配置需要提取的字段；list(string)
        save_path - 存储数据格式及保存路径的字典；string
        
    Returns:
        poisInAll_gdf - 提取给定字段的POI数据；GeoDataFrame（GeoPandas）

None
```
