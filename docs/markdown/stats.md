# usda.stats

?> 描述性统计

## stats.frequency_bins

频数分布计算

```python
frequency_bins(df, bins, field)
    function - 频数分布计算
    
    Params:
        df - 单列（数值类型）的DataFrame数据；DataFrame(Pandas)
        bins - 配置分割区间（组距）；range()，例如：range(0,600+50,50)
        field - 字段名；string
        
    Returns:
        df_fre - 统计结果字段包含：index（为bins）、fre、relFre、median和fre_percent%；DataFrame
```

## stats.comparisonOFdistribution

数据集z-score概率密度函数分布曲线（即观察值/实验值 observed/empirical data）与标准正态分布(即理论值 theoretical set)比较

```python
comparisonOFdistribution(df, field, bins=100)
    funciton-数据集z-score概率密度函数分布曲线（即观察值/实验值 observed/empirical data）与标准正态分布(即理论值 theoretical set)比较
    
    Params:
        df - 包含待分析数据集的DataFrame格式类型数据；DataFrame(Pandas)
        field - 指定分析数据数据（DataFrame格式）的列名；string
        bins - 指定频数宽度，为单一整数代表频数宽度（bin）的数量；或者列表，代表多个频数宽度的列表。The default is 100；int;list(int)
        
    Returns:
        None
```

?> 统计推断

## stats.is_outlier

判断异常值

```python
is_outlier(data, threshold=3.5)
    function-判断异常值
        
    Params:
        data - 待分析的数据，列表或者一维数组；list/array
        threshold - 判断是否为异常值的边界条件, The default is 3.5；float
        
    Returns
        is_outlier_bool - 判断异常值后的布尔值列表；list(bool)
        data[~is_outlier_bool] - 移除异常值后的数值列表；list
```

## stats.ptsKDE_geoDF2raster

计算GeoDaraFrame格式的点数据核密度估计，并转换为栅格数据

```python
ptsKDE_geoDF2raster(pts_geoDF, raster_path, cellSize, scale)
    function - 计算GeoDaraFrame格式的点数据核密度估计，并转换为栅格数据
    
    Params:
        pts_geoDF - GeoDaraFrame格式的点数据；GeoDataFrame(GeoPandas)
        raster_path - 保存的栅格文件路径；string
        cellSize - 栅格单元大小；int
        scale - 缩放核密度估计值；int/float
        
    Returns:
        返回读取已经保存的核密度估计栅格数据；array
```
