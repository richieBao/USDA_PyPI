# usda.charts

## charts.plotly_table

使用Plotly，以表格形式显示DataFrame格式数据

```python
plotly_table(df, column_extraction)
    funciton - 使用Plotly，以表格形式显示DataFrame格式数据
    
    Params:
        df - 输入的DataFrame或者GeoDataFrame；[Geo]DataFrame
        column_extraction - 提取的字段（列）；list(string)
    
    Returns:
        None
```

## charts.probability_graph

正态分布概率计算及图形表述

```python
probability_graph(x_i, x_min, x_max, x_s=-9999, left=True, step=0.001, subplot_num=221, loc=0, scale=1)
    function - 正态分布概率计算及图形表述
    
    Paras:
        x_i - 待预测概率的值；float
        x_min - 数据集区间最小值；float
        x_max - 数据集区间最大值；float
        x_s - 第2个带预测概率的值，其值大于x_i值。The default is -9999；float
        left - 是否计算小于或等于，或者大于或等于指定值的概率。The default is True；bool
        step - 数据集区间的步幅。The default is 0.001；float
        subplot_num - 打印子图的序号，例如221中，第一个2代表列，第二个2代表行，第三个是子图的序号，即总共2行2列总共4个子图，1为第一个子图。The default is 221；int
        loc - 即均值。The default is 0；float
        scale - 标准差。The default is 1；float
        
    Returns:
        None
```