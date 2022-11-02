# usda.data_visualization

## data_visualization.plotly_table

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

## data_visualization.probability_graph

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

## data_visualization.print_html

在Jupyter中打印DataFrame格式数据为HTML

```python
print_html(df, row_numbers=5)
    function - 在Jupyter中打印DataFrame格式数据为HTML
    
    Params:
        df - 需要打印的DataFrame或GeoDataFrame格式数据；DataFrame
        row_numbers - 打印的行数，如果为正，从开始打印如果为负，从末尾打印；int
        
    Returns:
        转换后的HTML格式数据；
```

## data_visualization.demo_con_style

在matplotlib的子图中绘制连接线。参考： matplotlib官网Connectionstyle Demo

```python
demo_con_style(a_coordi, b_coordi, ax, connectionstyle)
    function - 在matplotlib的子图中绘制连接线。参考： matplotlib官网Connectionstyle Demo
    
    Params:
        a_coordi - a点的x，y坐标；tuple
        b_coordi - b点的x，y坐标；tuple
        ax - 子图；ax(plot)
        connectionstyle - 连接线的形式；string
        
    Returns:
        None
```

##charts.generate_colors

生成颜色列表或者字典

```python
generate_colors()
    function - 生成颜色列表或者字典
    
    Returns:
        hex_colors_only - 16进制颜色值列表；list
        hex_colors_dic - 颜色名称：16进制颜色值；dict
        rgb_colors_dic - 颜色名称：(r,g,b)；dict
```

## data_visualization.demo_con_style_multiple

在matplotlib的子图中绘制多个连接线

```python
demo_con_style_multiple(a_coordi, b_coordi, ax, connectionstyle)
    function - 在matplotlib的子图中绘制多个连接线
    reference：matplotlib官网Connectionstyle Demo :https://matplotlib.org/3.3.2/gallery/userdemo/connectionstyle_demo.html#sphx-glr-gallery-userdemo-connectionstyle-demo-py
    
    Params:
        a_coordi - 起始点的x，y坐标；tuple
        b_coordi - 结束点的x，y坐标；tuple
        ax - 子图；ax(plot)
        connectionstyle - 连接线的形式；string
    
    Returns:
        None
```

## data_visualization.generate_colors

生成颜色列表或者字典

```python
generate_colors()
    function - 生成颜色列表或者字典
    
    Returns:
        hex_colors_only - 16进制颜色值列表；list
        hex_colors_dic - 颜色名称：16进制颜色值；dict
        rgb_colors_dic - 颜色名称：(r,g,b)；dict
```

## data_visualization.bands_show

指定波段，同时显示多个遥感影像

```python
bands_show(img_stack_list, band_num)
    function - 指定波段，同时显示多个遥感影像
    
    Params:
        img_stack_list - 影像列表；list(array)
        band_num - 显示的层列表；list(int)
```

## data_visualization.image_exposure

拉伸图像 contract stretching

```python
image_exposure(img_bands, percentile=(2, 98))
    function - 拉伸图像 contract stretching
    
    Params:
        img_bands - landsat stack后的波段；array
        percentile - 百分位数，The default is (2,98)；tuple
    
    Returns:
        img_bands_exposure - 返回拉伸后的影像；array
```

## data_visualization.downsampling_blockFreqency

降采样二维数组，根据每一block内值得频数最大值，即最多出现得值为每一block的采样值

```python
downsampling_blockFreqency(array_2d, blocksize=[10, 10])
    fuction - 降采样二维数组，根据每一block内值得频数最大值，即最多出现得值为每一block的采样值
    
    Params:
        array_2d - 待降采样的二维数组；array(2d)
        blocksize - block大小，即每一采用的范围，The default is [10,10]；tuple
        
    Returns:
        downsample - 降采样结果；array
```

## data_visualization.data_division

将数据按照给定的百分数划分，并给定固定的值，整数值或RGB色彩值

```python
data_division(data, division, right=True)
    function - 将数据按照给定的百分数划分，并给定固定的值，整数值或RGB色彩值
    
    Params:
        data - 待划分的numpy数组；array
        division - 百分数列表，例如[0,35,85]；list
        right - bool,optional. The default is True. 
                Indicating whether the intervals include the right or the left bin edge. 
                Default behavior is (right==False) indicating that the interval does not include the right edge. 
                The left bin end is open in this case, i.e., bins[i-1] <= x < bins[i] is the default behavior for monotonically increasing bins.        
    
    Returns：
        data_digitize - 返回整数值；list(int)
        data_rgb - 返回RGB，颜色值；list
```

## data_visualization.percentile_slider

多个栅格数据，给定百分比，变化观察

```python
percentile_slider(season_dic)
    function - 多个栅格数据，给定百分比，变化观察
    
    Params:
        season_dic -  多个栅格字典，键为自定义键名，值为读取的栅格数据（array），例如{"w_180310":w_180310_NDVI_rescaled,"s_190820":s_190820_NDVI_rescaled,"a_191018":a_191018_NDVI_rescaled}；dict
        
    Returns:
        None
```

## data_visualization.uniqueish_color

使用matplotlib提供的方法随机返回浮点型RGB

```python
uniqueish_color()
    function - 使用matplotlib提供的方法随机返回浮点型RGB
```

## data_visualization.img_struc_show

显示图像以及颜色R值，或G,B值

```python
img_struc_show(img_fp, val='R', figsize=(7, 7))
    function - 显示图像以及颜色R值，或G,B值
    
    Params:
        img_fp - 输入图像文件路径；string
        val - 选择显示值，R，G，或B，The default is 'R'；string
        figsize - 配置图像显示大小，The default is (7,7)；tuple
        
    Returns:
        None
```

## data_visualization.animated_gif_show

读入.gif，并动态显示

```python
animated_gif_show(gif_fp, figsize=(8, 8))
    function - 读入.gif，并动态显示
    
    Params:
        gif_fp - GIF文件路径；string
        figsize - 图表大小，The default is (8,8)；tuple
        
    Returns:
        HTML
```

## data_visualization.Gaussion_blur

应用OpenCV计算高斯模糊，并给定滑动条调节参数

```python
Gaussion_blur(img_fp)
    function - 应用OpenCV计算高斯模糊，并给定滑动条调节参数
    
    Params:
        img_fp - 图像路径；string
    
    Returns:
        None
```

## data_visualization.STAR_detection

使用Star特征检测器提取图像特征

```python
STAR_detection(img_fp, save=False)
    function - 使用Star特征检测器提取图像特征
    Params:
        img_fp - 图像文件路径  
        save- 是否保存特征结果图像。 The default is False；bool
        
    Returns:
        None
```

## data_visualization.plotly_scatterMapbox

使用plotly的go.Scattermapbox方法，在地图上显示点及其连线，坐标为经纬度

```python
plotly_scatterMapbox(df, **kwargs)
    function - 使用plotly的go.Scattermapbox方法，在地图上显示点及其连线，坐标为经纬度
    
    Paras:
        df - DataFrame格式数据，含经纬度；DataFrame
        field - 'lon':df的longitude列名，'lat'：为df的latitude列名，'center_lon':地图显示中心精经度定位，"center_lat":地图显示中心维度定位，"zoom"：为地图缩放；string
```

## data_visualization.DynamicStreetView_visualPerception

应用Star提取图像关键点，结合SIFT获得描述子，根据特征匹配分析特征变化（视觉感知变化），即动态街景视觉感知

```python
class DynamicStreetView_visualPerception(builtins.object)
 |  DynamicStreetView_visualPerception(imgs_fp, knnMatch_ratio=0.75)
 |  
 |  class - 应用Star提取图像关键点，结合SIFT获得描述子，根据特征匹配分析特征变化（视觉感知变化），即动态街景视觉感知
 |  
 |  Params:
 |      imgs_fp - 图像路径列表；list(string)
 |      knnMatch_ratio - 图像匹配比例，默认为0.75；float
 |  
 |  Methods defined here:
 |  
 |  __init__(self, imgs_fp, knnMatch_ratio=0.75)
 |      Initialize self.  See help(type(self)) for accurate signature.
 |  
 |  feature_matching(self, des_1, des_2, kp_1=None, kp_2=None)
 |      function - 图像匹配
 |  
 |  kp_descriptor(self, img_fp)
 |      function - 提取关键点和获取描述子
 |  
 |  sequence_statistics(self)
 |      function - 序列图像匹配计算，每一位置图像与后续所有位置匹配分析
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |  
 |  __dict__
 |      dictionary for instance variables (if defined)
 |  
 |  __weakref__
 |      list of weak references to the object (if defined)
```

## data_visualization.knee_lineGraph

绘制折线图，及其拐点。需调用kneed库的KneeLocator，及DataGenerator文件

```python
knee_lineGraph(x, y)
    function - 绘制折线图，及其拐点。需调用kneed库的KneeLocator，及DataGenerator文件
    
    Paras:
        x - 横坐标，用于横轴标签
        y - 纵坐标，用于计算拐点
```

## data_visualization.movingAverage_inflection

曲线（数据）平滑，与寻找曲线水平和纵向的斜率变化点

```python
class movingAverage_inflection(builtins.object)
 |  movingAverage_inflection(series, window, plot_intervals=False, scale=1.96, plot_anomalies=False, figsize=(15, 5), threshold=0)
 |  
 |  class - 曲线（数据）平滑，与寻找曲线水平和纵向的斜率变化点
 |  
 |  Params:
 |      series - pandas 的Series格式数据
 |      window - 滑动窗口大小，值越大，平滑程度越大
 |      plot_intervals - 是否打印置信区间，某人为False 
 |      scale - 偏差比例，默认为1.96, 
 |      plot_anomalies - 是否打印异常值，默认为False,
 |      figsize - 打印窗口大小，默认为(15,5),
 |      threshold - 拐点阈值，默认为0
 |  
 |  Methods defined here:
 |  
 |  __init__(self, series, window, plot_intervals=False, scale=1.96, plot_anomalies=False, figsize=(15, 5), threshold=0)
 |      Initialize self.  See help(type(self)) for accurate signature.
 |  
 |  apply_mask(self, mask, x, y)
 |  
 |  knee_elbow(self)
 |      function - 返回拐点的起末位置
 |  
 |  masks(self, vec)
 |      function - 寻找曲线水平和纵向的斜率变化，参考 https://stackoverflow.com/questions/47342447/find-locations-on-a-curve-where-the-slope-changes
 |  
 |  movingAverage(self)
 |  
 |  plot_movingAverage(self, inflection=False)
 |      function - 打印移动平衡/滑动窗口，及拐点
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |  
 |  __dict__
 |      dictionary for instance variables (if defined)
 |  
 |  __weakref__
 |      list of weak references to the object (if defined)
```

## data_visualization.vanishing_position_length

计算图像匹配特征点几乎无关联的距离，即对特定位置视觉随距离远去而感知消失的距离

```python
vanishing_position_length(matches_num, coordi_df, epsg, **kwargs)
    function - 计算图像匹配特征点几乎无关联的距离，即对特定位置视觉随距离远去而感知消失的距离
    
    Params:
        matches_num - 由类dynamicStreetView_visualPerception计算的特征关键点匹配数量
        coordi_df - 包含经纬度的DataFrame，其列名为：lon,lat
        **kwargs - 同类movingAverage_inflection配置参数
```

## data_visualization.Sentinel2_bandsComposite_show

Sentinel-2波段合成显示。需要deg2num(lat_deg, lon_deg, zoom)和centroid(bounds)函数

```python
Sentinel2_bandsComposite_show(RGB_bands, zoom=10, tilesize=512, figsize=(10, 10))
    function - Sentinel-2波段合成显示。需要deg2num(lat_deg, lon_deg, zoom)和centroid(bounds)函数
    
    Params:
        RGB_bands - 波段文件路径名字典，例如{"R":path_R,"G":path_G,"B":path_B}；dict
        zoom - zoom level缩放级别。The defalut is 10；int
        tilesize - 瓦片大小。The default is 512；int
        figsize- 打印图表大小。The default is (10,10)；tuple
        
    Returns:
        None
```

## data_visualization.markBoundaries_layoutShow

给定包含多个图像分割的一个数组，排布显示分割图像边界

```python
markBoundaries_layoutShow(segs_array, img, columns, titles, prefix, figsize=(15, 10))
    function - 给定包含多个图像分割的一个数组，排布显示分割图像边界。
    
    Paras:
        segs_array - 多个图像分割数组；ndarray
        img - 底图 ；ndarray
        columns - 列数；int
        titles - 子图标题；string
        figsize - 图表大小。The default is (15,10)；tuple
        
    Returns:
        None
```

## data_visualization.imgs_layoutShow

显示一个文件夹下所有图片，便于查看

```python
imgs_layoutShow(imgs_root, imgsFn_lst, columns, scale, figsize=(15, 10))
    function - 显示一个文件夹下所有图片，便于查看。
    
    Params:
        imgs_root - 图像所在根目录；string
        imgsFn_lst - 图像名列表；list(string)
        columns - 列数；int
        
    Returns:
        None
```

## data_visualization.imgs_layoutShow_FPList

显示一个文件夹下所有图片，便于查看

```python
imgs_layoutShow_FPList(imgs_fp_list, columns, scale, figsize=(15, 10))
    function - 显示一个文件夹下所有图片，便于查看。
    
    Params:
        imgs_fp_list - 图像文件路径名列表；list(string)
        columns - 显示列数；int
        scale - 调整图像大小比例因子；float
        figsize - 打印图表大小。The default is (15,10)；tuple(int)
        
    Returns:
        None
```

## data_visualization.img_rescale

读取与压缩图像，返回2维度数组

```python
img_rescale(img_path, scale)
    function - 读取与压缩图像，返回2维度数组
    
    Params:
        img_path - 待处理图像路径；lstring
        scale - 图像缩放比例因子；float
    
    Returns:
        img_3d - 返回三维图像数组；ndarray        
        img_2d - 返回二维图像数组；ndarray
```

## data_visualization.img_theme_color

聚类的方法提取图像主题色，并打印图像、聚类预测类的二维显示和主题色带

```python
img_theme_color(imgs_root, imgsFn_lst, columns, scale)
    function - 聚类的方法提取图像主题色，并打印图像、聚类预测类的二维显示和主题色带
    
    Params:
        imgs_root - 图像所在根目录；string
        imgsFn_lst - 图像名列表；list(string)
        columns - 列数；int    
        
    Returns:
        themes - 图像主题色；array
        pred - 预测的类标；array
```

## data_visualization.themeColor_impression

显示所有图像主题色，获取总体印象

```python
themeColor_impression(theme_color)
    function - 显示所有图像主题色，获取总体印象
    
    Params:
        theme_color - 主题色数组；array
        
    Returns:
        None
```

## data_visualization.imgs_compression_cv

显示所有图像主题色，获取总体印象

```python
themeColor_impression(theme_color)
    function - 显示所有图像主题色，获取总体印象
    
    Params:
        theme_color - 主题色数组；array
        
    Returns:
        None
```

## data_visualization.boxplot_custom

根据matplotlib库的箱型图打印方法，自定义箱型图可调整的打印样式

```python
boxplot_custom(data_dict, **args)
    根据matplotlib库的箱型图打印方法，自定义箱型图可调整的打印样式。 
    
    Parameters
    ----------
    data_dict : dict(list,numerical)
        字典结构形式的数据，键为横坐分类数据，值为数值列表.
    **args : keyword arguments
        可调整的箱型图样式参数包括['figsize',  'fontsize',  'frameOn',  'xlabel',  'ylabel',  'labelsize',  'tick_length',  'tick_width',  'tick_color',  'tick_direction',  'notch',  'sym',  'whisker_linestyle',  'whisker_linewidth',  'median_linewidth',  'median_capstyle'].
    
    Returns
    -------
    paras : dict
        样式更新后的参数值.
```

## data_visualization.SIFT_detection

尺度不变特征变换(scale invariant feature transform，SIFT)特征点检测

```python
SIFT_detection(img_fp, save=False)
    function - 尺度不变特征变换(scale invariant feature transform，SIFT)特征点检测
    
    Params:
        img_fp - 图像文件路径；string
        save- 是否保存特征结果图像。 The default is False；bool
        
    Returns:
        None
```

## data_visualization.feature_matching

尺度不变特征变换(scale invariant feature transform，SIFT)特征点检测

```python
SIFT_detection(img_fp, save=False)
    function - 尺度不变特征变换(scale invariant feature transform，SIFT)特征点检测
    
    Params:
        img_fp - 图像文件路径；string
        save- 是否保存特征结果图像。 The default is False；bool
        
    Returns:
        None
```

## data_visualization.segMasks_layoutShow

给定包含多个图像分割的一个数组，排布显示分割图像掩码

```python
segMasks_layoutShow(segs_array, columns, titles, prefix, cmap='prism', figsize=(20, 10))
    function - 给定包含多个图像分割的一个数组，排布显示分割图像掩码。
    
    Paras:
        segs_array - 多个图像分割数组；ndarray
        columns - 列数；int
        titles - 子图标题；string
        figsize - 图表大小。The default is (20,10)；tuple(int)
```


