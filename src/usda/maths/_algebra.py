# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 18:45:22 2022

@author: richie bao
"""
import random
import sympy

def vector_plot_2d(ax_2d,C,origin_vector,vector,color='r',label='vector',width=0.022):
    '''
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
    '''
    origin_vector_matrix=origin_vector.to_matrix(C)
    x=origin_vector_matrix.row(0)[0]
    y=origin_vector_matrix.row(1)[0]

    vector_matrix=vector.to_matrix(C)
    u=vector_matrix.row(0)[0]
    v=vector_matrix.row(1)[0]

    ax_2d.quiver(float(x),float(y),float(u),float(v),color=color,label=label,width=width)

def vector_plot_3d(ax_3d,C,origin_vector,vector,color='r',label='vector',arrow_length_ratio=0.1):
    '''
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
    '''
    
    origin_vector_matrix=origin_vector.to_matrix(C)
    x=origin_vector_matrix.row(0)[0]
    y=origin_vector_matrix.row(1)[0]
    z=origin_vector_matrix.row(2)[0]
    
    vector_matrix=vector.to_matrix(C)
    u=vector_matrix.row(0)[0]
    v=vector_matrix.row(1)[0]
    w=vector_matrix.row(2)[0]
    ax_3d.quiver(x,y,z,u,v,w,color=color,label=label,arrow_length_ratio=arrow_length_ratio)
    
def move_alongVectors(vector_list,coeffi_list,C,ax,):
    '''
    function - 给定向量，及对应系数，延向量绘制
    
    Params:
        vector_list - 向量列表，按移动顺序；list(vector)
        coeffi_list - 向量的系数，例如；list(tuple(symbol,float))
        C - SymPy下定义的坐标系；CoordSys3D()
        ax - 子图；ax(matplotlib)
        
    Returns:
        None
    '''
    
    colors=[color[0] for color in mcolors.TABLEAU_COLORS.items()]  # mcolors.BASE_COLORS, mcolors.TABLEAU_COLORS,mcolors.CSS4_COLORS
    colors__random_selection=random.sample(colors,len(vector_list)-1)
    v_accumulation=[]
    v_accumulation.append(vector_list[0])
    # 每个向量绘制以之前所有向量之和为起点
    for expr in vector_list[1:]:
        v_accumulation.append(expr+v_accumulation[-1])
    
    v_accumulation=v_accumulation[:-1]   
    for i in range(1,len(vector_list)):
        vector_plot_3d(ax,C,v_accumulation[i-1].subs(coeffi_list),vector_list[i].subs(coeffi_list),color=colors__random_selection[i-1],label='v_%s'%coeffi_list[i-1][0],arrow_length_ratio=0.2)
 
def vector2matrix_rref(v_list,C):
    '''
    function - 将向量集合转换为向量矩阵，并计算简化的行阶梯形矩阵
    
    Params:
        v_list - 向量列表；list(vector)
        C - sympy定义的坐标系统;CoordSys3D()

    Returns:
        v_matrix.T - 转换后的向量矩阵,即线性变换矩阵        
    '''    
    v_matrix=v_list[0].to_matrix(C)
    for v in v_list[1:]:
        v_temp=v.to_matrix(C)
        v_matrix=v_matrix.col_insert(-1,v_temp)
        
    print("_"*50)
    pprint(v_matrix.T.rref())
    return v_matrix.T

