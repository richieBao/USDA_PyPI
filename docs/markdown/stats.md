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