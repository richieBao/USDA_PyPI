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

?> regression

## stats.coefficient_of_determination

回归方程的决定系数

```python
coefficient_of_determination(observed_vals, predicted_vals)
    function - 回归方程的决定系数
    
    Params:
        observed_vals - 观测值（实测值）；list(float)
        predicted_vals - 预测值；list(float)
        
    Returns:
        R_square_a - 决定系数，由观测值和预测值计算获得；float
        R_square_b - 决定系数，由残差平方和和总平方和计算获得；float
```

## stats.ANOVA

简单线性回归方程-回归显著性检验（回归系数检验）

```python
ANOVA(observed_vals, predicted_vals, df_reg, df_res)
    function - 简单线性回归方程-回归显著性检验（回归系数检验）
    
    Params:
        observed_vals - 观测值（实测值）；list(float)
        predicted_vals - 预测值；list(float)
        
    Returns:
        None
```

## stats.confidenceInterval_estimator_LR

简单线性回归置信区间估计，及预测区间

```python
confidenceInterval_estimator_LR(x, sample_num, X, y, model, confidence=0.05)
    function - 简单线性回归置信区间估计，及预测区间
    
    Params:
        x - 自变量取值；float
        sample_num - 样本数量；int
        X - 样本数据集-自变量；list(float)
        y - 样本数据集-因变量；list(float)
        model -使用sklearn获取的线性回归模型；model
        confidence -  置信度。The default is 0.05" ；float
    
    Returns:
    
       CI - 置信区间；list(float)
```

## stats.correlationAnalysis_multivarialbe

DataFrame数据格式，成组计算pearsonr相关系数

```python
    function - DataFrame数据格式，成组计算pearsonr相关系数
    
    Params:
        df - DataFrame格式数据集；DataFrame(float)
    
    Returns:
        p_values - P值；DataFrame(float)
        correlation - 相关系数；DataFame(float)
```

## stats.coefficient_of_determination_correction

回归方程修正自由度的判定系数

```python
coefficient_of_determination_correction(observed_vals, predicted_vals, independent_variable_n)
    function - 回归方程修正自由度的判定系数
    
    Params:
        observed_vals - 观测值（实测值）；list(float)
        predicted_vals - 预测值；list(float)
        independent_variable_n - 自变量个数；int
        
    Returns:
        R_square_correction - 正自由度的判定系数；float
```

## stats.confidenceInterval_estimator_LR_multivariable

多元线性回归置信区间估计，及预测区间

```python
confidenceInterval_estimator_LR_multivariable(X, y, model, confidence=0.05)
    function - 多元线性回归置信区间估计，及预测区间
    
    Params:
        X - 样本自变量；DataFrame数据格式
        y - 样本因变量；list(float)
        model - 多元回归模型；model
        confidence - 置信度，The default is 0.05；float
    
    return:
        CI- 预测值的置信区间；list(float)
```

