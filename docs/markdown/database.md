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