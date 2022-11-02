# maths

## maths.vector_plot_2d

转换SymPy的vector及Matrix数据格式为matplotlib可以打印的数据格式

```python
vector_plot_2d(ax_2d, C, origin_vector, vector, color='r', label='vector', width=0.022)
    funciton - 转换SymPy的vector及Matrix数据格式为matplotlib可以打印的数据格式
    
    Params:
        ax_2d - matplotlib的2d格式子图；ax(matplotlib)
        C - /coordinate_system - SymPy下定义的坐标系；CoordSys3D()
        origin_vector - 如果是固定向量，给定向量的起点（使用向量，即表示从坐标原点所指向的位置），如果是自由向量，起点设置为坐标原点；vector(SymPy)
        vector - 所要打印的向量；vector(SymPy)
        color - 向量色彩，The default is 'r'；string
        label - 向量标签，The default is 'vector'；string
        arrow_length_ratio - 向量箭头大小，The default is 0.022；float 
        
    Returns:
        None
```

## maths.vector_plot_3d

转换SymPy的vector及Matrix数据格式为matplotlib可以打印的数据格式

```python
vector_plot_3d(ax_3d, C, origin_vector, vector, color='r', label='vector', arrow_length_ratio=0.1)
    funciton - 转换SymPy的vector及Matrix数据格式为matplotlib可以打印的数据格式
    
    Params:
        ax_3d - matplotlib的3d格式子图；ax(matplotlib)
        C - /coordinate_system - SymPy下定义的坐标系；CoordSys3D()
        origin_vector - 如果是固定向量，给定向量的起点（使用向量，即表示从坐标原点所指向的位置），如果是自由向量，起点设置为坐标原点；vector(SymPy)
        vector - 所要打印的向量；vector(SymPy)
        color - 向量色彩，The default is 'r'；string
        label - 向量标签，The default is 'vector'；string
        arrow_length_ratio - 向量箭头大小，The default is 0.1；float  
        
    Returns:
        None
```

## maths.move_alongVectors

给定向量，及对应系数，延向量绘制

```python
move_alongVectors(vector_list, coeffi_list, C, ax)
    function - 给定向量，及对应系数，延向量绘制
    
    Params:
        vector_list - 向量列表，按移动顺序；list(vector)
        coeffi_list - 向量的系数，例如；list(tuple(symbol,float))
        C - SymPy下定义的坐标系；CoordSys3D()
        ax - 子图；ax(matplotlib)
        
    Returns:
        None
```

## maths.vector2matrix_rref

将向量集合转换为向量矩阵，并计算简化的行阶梯形矩阵

```python
vector2matrix_rref(v_list, C)
    function - 将向量集合转换为向量矩阵，并计算简化的行阶梯形矩阵
    
    Params:
        v_list - 向量列表；list(vector)
        C - sympy定义的坐标系统;CoordSys3D()
    
    Returns:
        v_matrix.T - 转换后的向量矩阵,即线性变换矩阵
```

## maths.circle_lines

给定圆心，半径，划分份数，计算所有直径的首尾坐标

```python
circle_lines(center, radius, division)
    function - 给定圆心，半径，划分份数，计算所有直径的首尾坐标
    
    Params:
        center - 圆心，例如(0,0)；tuple
        radius - 半径；float
        division - 划分份数；int
        
    Returns:
        xy - 首坐标数组；array
        xy_ -尾坐标数组；array
        xy_head_tail - 收尾坐标数组；array
```

## maths.point_Proj2Line

计算二维点到直线上的投影

```python
point_Proj2Line(line_endpts, point)
    function - 计算二维点到直线上的投影
    
    Params:
        line_endpts - 直线首尾点坐标，例如((2,0),(-2,0))；tuple/list
        point - 待要投影的点，例如[-0.11453985,  1.23781631]；tuple/list
    
    Returns:
        P - 投影点；tuple/list
```