# models

## models.k_neighbors_entire

返回指定邻近数目的最近点坐标

```python
k_neighbors_entire(xy, k=3)
    function - 返回指定邻近数目的最近点坐标
    
    Params:
        xy - 点坐标二维数组，例如
            array([[23714. ,   364.7],
                  [21375. ,   331.4],
                  [32355. ,   353.7],
                  [35503. ,   273.3]]
        k - 指定邻近数目；int
    
    return:
        neighbors - 返回各个点索引，以及各个点所有指定数目邻近点索引
```

## models.PolynomialFeatures_regularization

多项式回归degree次数选择，及正则化

```python
PolynomialFeatures_regularization(X, y, regularization='linear')
    function - 多项式回归degree次数选择，及正则化
    
    Params:
        X - 解释变量；array
        y - 响应变量；array
        regularization - 正则化方法， 为'linear'时，不进行正则化，正则化方法为'Ridge'和'LASSO'；string
        
    Returns:
        reg - model
```

## models.dim1_convolution_SubplotAnimation

一维卷积动画解析，可以自定义系统函数和信号函数

```python
class dim1_convolution_SubplotAnimation(matplotlib.animation.TimedAnimation)
 |  dim1_convolution_SubplotAnimation(G_T_fun, F_T_fun, t={'s': -10, 'e': 10, 'step': 1, 'linespace': 1000}, mode='same')
 |  
 |  function - 一维卷积动画解析，可以自定义系统函数和信号函数   
 |  
 |  Params:
 |      G_T_fun - 系统响应函数；func
 |      F_T_fun - 输入信号函数；func
 |      t={"s":-10,"e":10,'step':1,'linespace':1000} -  时间开始点、结束点、帧的步幅、时间段细分；dict
 |      mode='same' - numpy库提供的convolve卷积方法的卷积模式；string
 |  
 |  Method resolution order:
 |      dim1_convolution_SubplotAnimation
 |      matplotlib.animation.TimedAnimation
 |      matplotlib.animation.Animation
 |      builtins.object
 |  
 |  Methods defined here:
 |  
 |  __init__(self, G_T_fun, F_T_fun, t={'s': -10, 'e': 10, 'step': 1, 'linespace': 1000}, mode='same')
 |      Initialize self.  See help(type(self)) for accurate signature.
 |  
 |  new_frame_seq(self)
 |      Return a new sequence of frame information.
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from matplotlib.animation.Animation:
 |  
 |  __del__(self)
 |  
 |  new_saved_frame_seq(self)
 |      Return a new sequence of saved/cached frame information.
 |  
 |  pause(self)
 |      Pause the animation.
 |  
 |  resume(self)
 |      Resume the animation.
 |  
 |  save(self, filename, writer=None, fps=None, dpi=None, codec=None, bitrate=None, extra_args=None, metadata=None, extra_anim=None, savefig_kwargs=None, *, progress_callback=None)
 |      Save the animation as a movie file by drawing every frame.
 |      
 |      Parameters
 |      ----------
 |      filename : str
 |          The output filename, e.g., :file:`mymovie.mp4`.
 |      
 |      writer : `MovieWriter` or str, default: :rc:`animation.writer`
 |          A `MovieWriter` instance to use or a key that identifies a
 |          class to use, such as 'ffmpeg'.
 |      
 |      fps : int, optional
 |          Movie frame rate (per second).  If not set, the frame rate from the
 |          animation's frame interval.
 |      
 |      dpi : float, default: :rc:`savefig.dpi`
 |          Controls the dots per inch for the movie frames.  Together with
 |          the figure's size in inches, this controls the size of the movie.
 |      
 |      codec : str, default: :rc:`animation.codec`.
 |          The video codec to use.  Not all codecs are supported by a given
 |          `MovieWriter`.
 |      
 |      bitrate : int, default: :rc:`animation.bitrate`
 |          The bitrate of the movie, in kilobits per second.  Higher values
 |          means higher quality movies, but increase the file size.  A value
 |          of -1 lets the underlying movie encoder select the bitrate.
 |      
 |      extra_args : list of str or None, optional
 |          Extra command-line arguments passed to the underlying movie
 |          encoder.  The default, None, means to use
 |          :rc:`animation.[name-of-encoder]_args` for the builtin writers.
 |      
 |      metadata : dict[str, str], default: {}
 |          Dictionary of keys and values for metadata to include in
 |          the output file. Some keys that may be of use include:
 |          title, artist, genre, subject, copyright, srcform, comment.
 |      
 |      extra_anim : list, default: []
 |          Additional `Animation` objects that should be included
 |          in the saved movie file. These need to be from the same
 |          `matplotlib.figure.Figure` instance. Also, animation frames will
 |          just be simply combined, so there should be a 1:1 correspondence
 |          between the frames from the different animations.
 |      
 |      savefig_kwargs : dict, default: {}
 |          Keyword arguments passed to each `~.Figure.savefig` call used to
 |          save the individual frames.
 |      
 |      progress_callback : function, optional
 |          A callback function that will be called for every frame to notify
 |          the saving progress. It must have the signature ::
 |      
 |              def func(current_frame: int, total_frames: int) -> Any
 |      
 |          where *current_frame* is the current frame number and
 |          *total_frames* is the total number of frames to be saved.
 |          *total_frames* is set to None, if the total number of frames can
 |          not be determined. Return values may exist but are ignored.
 |      
 |          Example code to write the progress to stdout::
 |      
 |              progress_callback =                    lambda i, n: print(f'Saving frame {i} of {n}')
 |      
 |      Notes
 |      -----
 |      *fps*, *codec*, *bitrate*, *extra_args* and *metadata* are used to
 |      construct a `.MovieWriter` instance and can only be passed if
 |      *writer* is a string.  If they are passed as non-*None* and *writer*
 |      is a `.MovieWriter`, a `RuntimeError` will be raised.
 |  
 |  to_html5_video(self, embed_limit=None)
 |      Convert the animation to an HTML5 ``<video>`` tag.
 |      
 |      This saves the animation as an h264 video, encoded in base64
 |      directly into the HTML5 video tag. This respects :rc:`animation.writer`
 |      and :rc:`animation.bitrate`. This also makes use of the
 |      ``interval`` to control the speed, and uses the ``repeat``
 |      parameter to decide whether to loop.
 |      
 |      Parameters
 |      ----------
 |      embed_limit : float, optional
 |          Limit, in MB, of the returned animation. No animation is created
 |          if the limit is exceeded.
 |          Defaults to :rc:`animation.embed_limit` = 20.0.
 |      
 |      Returns
 |      -------
 |      str
 |          An HTML5 video tag with the animation embedded as base64 encoded
 |          h264 video.
 |          If the *embed_limit* is exceeded, this returns the string
 |          "Video too large to embed."
 |  
 |  to_jshtml(self, fps=None, embed_frames=True, default_mode=None)
 |      Generate HTML representation of the animation.
 |      
 |      Parameters
 |      ----------
 |      fps : int, optional
 |          Movie frame rate (per second). If not set, the frame rate from
 |          the animation's frame interval.
 |      embed_frames : bool, optional
 |      default_mode : str, optional
 |          What to do when the animation ends. Must be one of ``{'loop',
 |          'once', 'reflect'}``. Defaults to ``'loop'`` if ``self.repeat``
 |          is True, otherwise ``'once'``.
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors inherited from matplotlib.animation.Animation:
 |  
 |  __dict__
 |      dictionary for instance variables (if defined)
 |  
 |  __weakref__
 |      list of weak references to the object (if defined)
```

## models.G_T_type_1

定义系统响应函数.类型-1

```python
G_T_type_1()
    function - 定义系统响应函数.类型-1
    
    Returns:
        g_t_ - sympy定义的函数
```

## models.F_T_type_1

定义输入信号函数，类型-1

```python
F_T_type_1(timing)
    function - 定义输入信号函数，类型-1
    
    return:
        函数计算公式
```

## models.G_T_type_2

定义系统响应函数.类型-2

```python
G_T_type_2()
    function - 定义系统响应函数.类型-2
    
    return:
        g_t_ - sympy定义的函数
```

## models.F_T_type_2

定义输入信号函数，类型-2

```python
F_T_type_2(timing)
    function - 定义输入信号函数，类型-2
    
    return:
        函数计算公式
```

## models.curve_segmentation_1DConvolution

应用一维卷积，根据跳变点分割数据

```python
curve_segmentation_1DConvolution(data, threshold=1)
    function - 应用一维卷积，根据跳变点分割数据
    
    Params:
        data - 待处理的一维度数据；list/array
    
    Returns:
        data_seg - 列表分割字典，"dataIdx_jump"-分割索引值，"data_jump"-分割原始数据，"conv_jump"-分割卷积结果
```

## models.SIR_deriv

定义SIR传播模型微分方程

```python
SIR_deriv(y, t, N, beta, gamma, plot=False)
    function - 定义SIR传播模型微分方程
    
    Params:
        y - S,I,R初始化值（例如，人口数）；tuple
        t - 时间序列；list
        N - 总人口数；int
        beta - 易感人群到受感人群转化比例；float
        gamma - 受感人群到恢复人群转换比例；float
    
    Rreturns:
        SIR_array - S, I, R数量；array
        
    Examples
    --------
    N=1000 # 总人口数
    I_0,R_0=1,0 # 初始化受感人群，及恢复人群的人口数
    S_0=N-I_0-R_0 # 有受感人群和恢复人群，计算得易感人群人口数
    beta,gamma=0.2,1./10 # 配置参数b(即beta)和k(即gamma)
    t=np.linspace(0,160,160) # 配置时间序列    
    y_0=S_0,I_0,R_0
    SIR_array=SIR_deriv(y_0,t,N,beta,gamma,plot=True)
```

## models.convolution_diffusion_img

定义基于SIR模型的二维卷积扩散

```python
class convolution_diffusion_img(builtins.object)
 |  convolution_diffusion_img(img_path, save_path, hours_per_second, dt, fps)
 |  
 |  class - 定义基于SIR模型的二维卷积扩散
 |  
 |  Parasm:
 |      img_path - 图像文件路径；string
 |      save_path - 保持的.gif文件路径；string
 |      hours_per_second - 扩散时间长度；int
 |      dt - 时间记录值，开始值；int
 |      fps - 配置moviepy，write_gif写入GIF每秒帧数；int 
 |  
 |  Examples
 |  --------
 |  img_12Pix_fp=r'./data/12mul12Pixel_1red.bmp' # 图像文件路径
 |  SIRSave_fp=r'./data/12mul12Pixel_1red_SIR.gif'
 |  hours_per_second=20
 |  dt=1 # 时间记录值，开始值
 |  fps=15 # 配置moviepy，write_gif写入GIF每秒帧数
 |  convDiff_img=convolution_diffusion_img(img_path=img_12Pix_fp,save_path=SIRSave_fp,hours_per_second=hours_per_second,dt=dt,fps=fps)
 |  convDiff_img.execute_()
 |  
 |  Methods defined here:
 |  
 |  __init__(self, img_path, save_path, hours_per_second, dt, fps)
 |      Initialize self.  See help(type(self)) for accurate signature.
 |  
 |  dispersion(self, SIR, dispersion_kernel)
 |      卷积扩散
 |  
 |  execute_(self)
 |      执行程序
 |  
 |  make_frame(self, t)
 |      返回每一步卷积的数据到VideoClip中
 |  
 |  update(self, world)
 |      更新数组，即基于前一步卷积结果的每一步卷积
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

## models.superpixel_segmentation_Felzenszwalb

像素分割，skimage库felzenszwalb方法。给定scale参数列表，批量计算

```python
superpixel_segmentation_Felzenszwalb(img, scale_list, sigma=0.5, min_size=50)
    function - 超像素分割，skimage库felzenszwalb方法。给定scale参数列表，批量计算
    
    Params:
        img - 读取的遥感影像、图像；ndarray
        scale_list - 分割比例列表；list(float)
        sigma - Width (standard deviation) of Gaussian kernel used in preprocessing.The default is 0.5； float
        min_size - Minimum component size. Enforced using postprocessing. The default is 50； int
        
    Returns:
        分割结果。Integer mask indicating segment labels；ndarray
```

## models.superpixel_segmentation_quickshift

超像素分割，skimage库quickshift方法。给定kernel_size参数列表，批量计算

```python
superpixel_segmentation_quickshift(img, kernel_sizes, ratio=0.5)
    function - 超像素分割，skimage库quickshift方法。给定kernel_size参数列表，批量计算
    
    Params:
        img - Input image. The axis corresponding to color channels can be specified via the channel_axis argument；ndarray
        kernel_sizes - Width of Gaussian kernel used in smoothing the sample density. Higher means fewer clusters；float, optional
        ratio - Balances color-space proximity and image-space proximity. Higher values give more weight to color-space. The default is 0.5；float, optional, between 0 and 1
        
    Returns:
        Integer mask indicating segment labels.
```

## models.multiSegs_stackStatistics

多尺度超像素级分割结果叠合频数统计

```python
multiSegs_stackStatistics(segs, save_fp)
    function - 多尺度超像素级分割结果叠合频数统计
    
    Params:
        segs - 超级像素分割结果。Integer mask indicating segment labels；ndarray（int）
        save_fp - 保存路径名（pickle）；string
        
    Returns:
        stack_statistics - 统计结果字典；dict
```

## models.feature_builder_BOW

根据所有图像关键点描述子聚类建立图像视觉词袋，获取每一图像的特征（码本）映射的频数统计

```python
class feature_builder_BOW(builtins.object)
 |  feature_builder_BOW(num_cluster=32)
 |  
 |  class - 根据所有图像关键点描述子聚类建立图像视觉词袋，获取每一图像的特征（码本）映射的频数统计
 |  
 |  Methods defined here:
 |  
 |  __init__(self, num_cluster=32)
 |      Initialize self.  See help(type(self)) for accurate signature.
 |  
 |  construct_feature(self, img, kmeans)
 |      function - 使用聚类的视觉词袋构建图像特征（构造码本）
 |      
 |      Paras:
 |          img - 读取的单张图像
 |          kmeans - 已训练的聚类模型
 |  
 |  extract_features(self, img)
 |      function - 提取图像特征
 |      
 |      Params:
 |          img - 读取的图像
 |  
 |  get_feature_map(self, training_data, kmeans)
 |      function - 返回每个图像的特征映射（码本映射）
 |      
 |      Paras:
 |          training_data - 训练数据集
 |          kmeans - 已训练的聚类模型
 |  
 |  get_visual_BOW(self, training_data)
 |      function - 提取图像特征，返回所有图像关键点聚类视觉词袋
 |      
 |      Params:
 |          training_data - 训练数据集
 |  
 |  normalize(self, input_data)
 |      fuction - 归一化数据
 |      
 |      Params:
 |          input_data - 待归一化的数组
 |  
 |  visual_BOW(self, des_all)
 |      function - 聚类所有图像的特征（描述子/SIFT），建立视觉词袋
 |      
 |      Params:
 |          des_all - 所有图像的关键点描述子
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

## models.df_multiColumns_LabelEncoder

根据指定的（多个）列，将分类转换为整数表示，区间为[0,分类数-1]

```python
df_multiColumns_LabelEncoder(df, columns=None)
    function - 根据指定的（多个）列，将分类转换为整数表示，区间为[0,分类数-1]
    
    Params:
        df - DataFrame格式数据；DataFrame
        columns - 指定待转换的列名列表；list(string)
        
    Returns:
        output - 分类整数编码；DataFrame
```

## models.entropy_compomnent

计算信息熵分量

```python
entropy_compomnent(numerator, denominator)
    function - 计算信息熵分量
    
    Params:
        numerator - 分子；
        denominator - 分母；
        
    Returns:
        信息熵分量；float
```

## models.IG

计算信息增量（IG）

```python
IG(df_dummies)
    function - 计算信息增量（IG）
    
    Params:
        df_dummies - DataFrame格式，独热编码的特征值；DataFrame
        
    Returns:
        cal_info_df - 信息增益（Information gain）；DataFrame
```

## models.decisionTree_structure

使用决策树分类，并打印决策树流程图表。迁移于Sklearn的'Understanding the decision tree structure', https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py

```python
decisionTree_structure(X, y, criterion='entropy', cv=None, figsize=(6, 6))
    function - 使用决策树分类，并打印决策树流程图表。迁移于Sklearn的'Understanding the decision tree structure', https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py
    
    Params:
        X - 数据集-特征值（解释变量）；ndarray
        y- 数据集-类标/标签(响应变量)；ndarray
        criterion - DecisionTreeClassifier 参数，衡量拆分的质量，即衡量哪一项检测最能减少分类的不确定性；string
        cv - cross_val_score参数，确定交叉验证分割策略，默认值为None，即5-fole(折)的交叉验证；int
        
    Returns:
        clf - 返回决策树模型
```

## models.ERF_trainer

用极端随机森林训练图像分类器

```python
class ERF_trainer(builtins.object)
 |  ERF_trainer(X, label_words, save_path)
 |  
 |  class - 用极端随机森林训练图像分类器
 |  
 |  Methods defined here:
 |  
 |  __init__(self, X, label_words, save_path)
 |      Initialize self.  See help(type(self)) for accurate signature.
 |  
 |  classify(self, X)
 |      function - 对未知数据的预测分类
 |  
 |  encode_labels(self, label_words)
 |      function - 对标签编码，及训练分类器
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

## models.ImageTag_extractor

图像识别器，基于图像分类模型，视觉词袋以及图像特征

```python
class ImageTag_extractor(builtins.object)
 |  ImageTag_extractor(ERF_clf_fp, visual_BOW_fp, visual_feature_fp)
 |  
 |  class - 图像识别器，基于图像分类模型，视觉词袋以及图像特征
 |  
 |  Methods defined here:
 |  
 |  __init__(self, ERF_clf_fp, visual_BOW_fp, visual_feature_fp)
 |      Initialize self.  See help(type(self)) for accurate signature.
 |  
 |  predict(self, img)
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

## models.SIR_spatialPropagating

SIR的空间传播模型

```python
class SIR_spatialPropagating(builtins.object)
 |  SIR_spatialPropagating(classi_array, cost_mapping, start_pt=[10, 10], beta=0.3, gamma=0.1, dispersion_rates=[0, 0.07, 0.03], dt=1.0, hours_per_second=168, duration=12, fps=15, SIR_gif_savePath='./SIR_sp.gif')
 |  
 |  funciton - SIR的空间传播模型
 |  
 |  Params:
 |      classi_array - 分类数据（.tif，或者其它图像类型），或者其它可用于成本计算的数据类型
 |      cost_mapping - 分类数据对应的成本值映射字典
 |      beta - beta值，确定S-->I的转换率
 |      gamma - gamma值，确定I-->R的转换率
 |      dispersion_rates - SIR三层栅格各自对应的卷积扩散率
 |      dt - 时间更新速度
 |      hours_per_second - 扩散时间长度/终止值(条件)
 |      duration - moviepy参数配置，持续时长
 |      fps - moviepy参数配置，每秒帧数
 |      SIR_gif_savePath - SIR空间传播计算结果.gif文件保存路径
 |      
 |  Examples
 |  --------
 |  # 成本栅格（数组）
 |  classi_array=mosaic_classi_array_rescaled    
 |  
 |  # 配置用地类型的成本值（空间阻力值）
 |  cost_H=250
 |  cost_M=125
 |  cost_L=50
 |  cost_Z=0
 |  cost_mapping={
 |              'never classified':(0,cost_Z),
 |              'unassigned':(1,cost_Z),
 |              'ground':(2,cost_M),
 |              'low vegetation':(3,cost_H),
 |              'medium vegetation':(4,cost_H),
 |              'high vegetation':(5,cost_H),
 |              'building':(6,cost_Z),
 |              'low point':(7,cost_Z),
 |              'reserved':(8,cost_M),
 |              'water':(9,cost_M),
 |              'rail':(10,cost_L),
 |              'road surface':(11,cost_L),
 |              'reserved':(12,cost_M),
 |              'wire-guard(shield)':(13,cost_M),
 |              'wire-conductor(phase)':(14,cost_M),
 |              'transimission':(15,cost_M),
 |              'wire-structure connector(insulator)':(16,cost_M),
 |              'bridge deck':(17,cost_L),
 |              'high noise':(18,cost_Z),
 |              'null':(9999,cost_Z)       
 |              }    
 |  
 |  # 参数配置
 |  start_pt=[418,640]  # [3724,3415]
 |  beta=0.3
 |  gamma=0.1
 |  dispersion_rates=[0, 0.07, 0.03]  # S层卷积扩散为0，I层卷积扩散为0.07，R层卷积扩散为0.03
 |  dt=1.0
 |  hours_per_second=30*24 # 7*24
 |  duration=12 #12
 |  fps=15 # 15
 |  SIR_gif_savePath=r"./imgs/SIR_sp.gif"
 |  
 |  SIR_sp=SIR_spatialPropagating(classi_array=classi_array,cost_mapping=cost_mapping,start_pt=start_pt,beta=beta,gamma=gamma,dispersion_rates=dispersion_rates,dt=dt,hours_per_second=hours_per_second,duration=duration,fps=fps,SIR_gif_savePath=SIR_gif_savePath)
 |  SIR_sp.execute()
 |  
 |  Methods defined here:
 |  
 |  __init__(self, classi_array, cost_mapping, start_pt=[10, 10], beta=0.3, gamma=0.1, dispersion_rates=[0, 0.07, 0.03], dt=1.0, hours_per_second=168, duration=12, fps=15, SIR_gif_savePath='./SIR_sp.gif')
 |      Initialize self.  See help(type(self)) for accurate signature.
 |  
 |  deriv(self, SIR, beta, gamma)
 |      SIR模型
 |  
 |  dispersion(self, SIR, dispersion_kernel, dispersion_rates)
 |      卷积扩散
 |  
 |  execute(self)
 |      执行程序
 |  
 |  make_frame(self, t)
 |      返回每一步的SIR和卷积综合蔓延结果
 |  
 |  update(self, world)
 |      执行SIR模型和卷积，更新world字典
 |  
 |  world_to_npimage(self, world)
 |      将模拟计算的值转换到[0,255]RGB色域空间
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

