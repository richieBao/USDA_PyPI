# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 16:13:49 2022

@author: richie bao 
"""
from IPython.display import HTML
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import math
import sympy
from sympy import pprint,Piecewise

class dim1_convolution_SubplotAnimation(animation.TimedAnimation):
    '''
    function - 一维卷积动画解析，可以自定义系统函数和信号函数   
    
    Params:
        G_T_fun - 系统响应函数；func
        F_T_fun - 输入信号函数；func
        t={"s":-10,"e":10,'step':1,'linespace':1000} -  时间开始点、结束点、帧的步幅、时间段细分；dict
        mode='same' - numpy库提供的convolve卷积方法的卷积模式；string
    '''    
    def __init__(self,G_T_fun,F_T_fun,t={"s":-10,"e":10,'step':1,'linespace':1000},mode='same'):  
        self.mode=mode
        self.start=t['s']
        self.end=t['e']
        self.step=t['step']
        self.linespace=t['linespace']
        self.t=np.linspace(self.start, self.end, self.linespace, endpoint=True)
        
        fig, axs=plt.subplots(1,3,figsize=(24,3))
        # 定义g(t)，系统响应函数
        self.g_t_=G_T_fun        
        g_t_val=self.g_t_(self.t)
        self.g_t_graph,=axs[0].plot(self.t,g_t_val,'--',label='g(t)',color='r')        
        
        # 定义f(t)，输入信号函数
        self.f_t_=F_T_fun
        self.f_t_graph,=axs[1].plot([],[],'-',label='f(t)',color='b')
        
        # 卷积（动态-随时间变化）
        self.convolution_graph,=axs[2].plot([],[],'-',label='1D convolution',color='g')

        axs[0].set_title('g_t')
        axs[0].legend(loc='lower left', frameon=False)
        axs[0].set_xlim(self.start,self.end)
        axs[0].set_ylim(-1.2,1.2)                
        
        axs[1].set_title('f_t')
        axs[1].legend(loc='lower left', frameon=False)
        axs[1].set_xlim(self.start,self.end)
        axs[1].set_ylim(-1.2,1.2)
        
        axs[2].set_title('1D convolution')
        axs[2].legend(loc='lower left', frameon=False)
        axs[2].set_xlim(self.start,self.end)
        axs[2].set_ylim(-1.2*100,1.2*100)

        plt.tight_layout()
        animation.TimedAnimation.__init__(self, fig, interval=500, blit=True) # interval配置更新速度        
           
    # 更新图形
    def _draw_frame(self,framedata):            
        i=framedata      
        f_t_sym=self.f_t_(i) # 1-先输入外部定义的F_T_fun函数的输入参数
        f_t_val=f_t_sym(self.t) # 2-再定义F_T_fun函数内部由sympy定义的公式的输入参数
        
        self.f_t_graph.set_data(self.t,f_t_val)   
        g_t_val=self.g_t_(self.t)
        g_t_val=g_t_val[~np.isnan(g_t_val)] # 移除空值，仅保留用于卷积部分的数据       
        
        if self.mode=='same':           
            conv=np.convolve(g_t_val,f_t_val,'same') # self.g_t_(t)
            self.convolution_graph.set_data(self.t,conv)            
            
        elif self.mode=='full':
            conv_=np.convolve(g_t_val,f_t_val,'full') # self.g_t_(t)
            t_diff=math.ceil((len(conv_)-len(self.t))/2)
            conv=conv_[t_diff:-t_diff+1]
            self.convolution_graph.set_data(self.t,conv)
            
        else:
            print("please define the mode value--'full' or 'same' ")
        
    # 配置帧frames    
    def new_frame_seq(self):
        return iter(np.arange(self.start,self.end,self.step))
    
    # 初始化图形
    def _init_draw(self):     
        graphs=[self.f_t_graph,self.convolution_graph,]
        for G in graphs:
            G.set_data([],[])        
            
def G_T_type_1():
    '''
    function - 定义系统响应函数.类型-1
    
    Returns:
        g_t_ - sympy定义的函数
    '''
    import sympy
    from sympy import pprint,Piecewise
    
    t,t_=sympy.symbols('t t_')
    g_t=1
    g_t_piecewise=Piecewise((g_t,(t>=0)&(t<=1))) # 定义位分段函数，系统响应函数在区间[0,1]之间作用。
    g_t_=sympy.lambdify(t,g_t_piecewise,"numpy")
    
    return g_t_

def F_T_type_1(timing):
    '''
    function - 定义输入信号函数，类型-1

    return:
        函数计算公式
    '''
    import sympy
    from sympy import pprint,Piecewise
    
    t,t_=sympy.symbols('t t_')
    f_t=1
    f_t_piecewise=Piecewise((f_t,(t>timing)&(t<timing+1)),(0,True)) # 定义位分段函数，系统响应函数在区间[0,1]之间作用
    f_t_=sympy.lambdify(t,f_t_piecewise,"numpy")
    
    return f_t_  

def G_T_type_2():
    '''
    function - 定义系统响应函数.类型-2
    
    return:
        g_t_ - sympy定义的函数
    '''
    import sympy
    from sympy import pprint,Piecewise
    
    t,t_=sympy.symbols('t t_')
    e=sympy.E
    g_t=-1*((e**t-e**(-t))/(e**t+e**(-t)))+1 # 参考Hyperbolic tangent function 即双曲正切函数  y=tanh x
    g_t_piecewise=Piecewise((g_t,(t>=0)&(t<3))) # 定义位分段函数，系统响应函数在区间[0,3]之间作用
    g_t_=sympy.lambdify(t,g_t_piecewise,"numpy")

    return g_t_

def F_T_type_2(timing):
    '''
    function - 定义输入信号函数，类型-2

    return:
        函数计算公式
    '''
    import sympy
    from sympy import pprint,Piecewise
    
    t,t_=sympy.symbols('t t_')
    f_t=1
    f_t_piecewise=Piecewise((f_t,(t>timing)&(t<timing+1)) ,(f_t,(t>timing-2)&(t<timing-1)) ,(0,True)) # 定义位分段函数，系统响应函数在区间[0,1]之间作用
    f_t_=sympy.lambdify(t,f_t_piecewise,"numpy")

    return f_t_              