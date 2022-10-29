# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 18:14:50 2022

@author: richie bao
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt   

from skimage import io
from scipy.ndimage.filters import convolve
import moviepy.editor as mpy

from sklearn.preprocessing import MinMaxScaler


# 定义SIR模型微分方程函数
def SIR_deriv(y,t,N,beta,gamma,plot=False):   
    '''
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
    '''

    def deriv(y,t,N,beta,gamma):
        S,I,R=y
        dSdt=-beta*S*I/N
        dIdt=beta*S*I/N-gamma*I
        dRdt=gamma*I
        
        return dSdt,dIdt,dRdt
    
    deriv_integration=odeint(deriv,y,t,args=(N,beta,gamma))
    S,I,R=deriv_integration.T
    SIR_array=np.stack([S,I,R])

    if plot==True:
        fig=plt.figure(facecolor='w',figsize=(12,6))
        ax=fig.add_subplot(111,facecolor='#dddddd',axisbelow=True)
        ax.plot(t,S/N,'b',alpha=0.5,lw=2,label='Susceptible')
        ax.plot(t,I/N,'r',alpha=0.5,lw=2,label='Infected')
        ax.plot(t,R/N,'g',alpha=0.5,lw=2,label='Recovered')
        
        ax.set_label('Time/days')
        ax.set_ylim(0,1.2)
        ax.yaxis.set_tick_params(length=0)
        ax.xaxis.set_tick_params(length=0)
        ax.grid(visible=True, which='major', c='w', lw=1, ls='-')
        legend = ax.legend()
        legend.get_frame().set_alpha(0.5)
        for spine in ('top', 'right', 'bottom', 'left'):
            ax.spines[spine].set_visible(False)    
        plt.show()   
    
    return SIR_array

class convolution_diffusion_img:
    '''
    class - 定义基于SIR模型的二维卷积扩散
    
    Parasm:
        img_path - 图像文件路径；string
        save_path - 保持的.gif文件路径；string
        hours_per_second - 扩散时间长度；int
        dt - 时间记录值，开始值；int
        fps - 配置moviepy，write_gif写入GIF每秒帧数；int 

    Examples
    --------
    img_12Pix_fp=r'./data/12mul12Pixel_1red.bmp' # 图像文件路径
    SIRSave_fp=r'./data/12mul12Pixel_1red_SIR.gif'
    hours_per_second=20
    dt=1 # 时间记录值，开始值
    fps=15 # 配置moviepy，write_gif写入GIF每秒帧数
    convDiff_img=convolution_diffusion_img(img_path=img_12Pix_fp,save_path=SIRSave_fp,hours_per_second=hours_per_second,dt=dt,fps=fps)
    convDiff_img.execute_()                
    '''
    def __init__(self,img_path,save_path,hours_per_second,dt,fps):  
        self.save_path=save_path
        self.hours_per_second=hours_per_second
        self.dt=dt
        self.fps=fps        
        img=io.imread(img_path)
        SIR=np.zeros((1,img.shape[0], img.shape[1]),dtype=np.int32) # 在配置SIR数组时，为三维度，是为了对接后续物种散步程序对SIR的配置
        SIR[0]=img.T[0] # 将图像的RGB中R通道值赋值给SIR
        self.world={'SIR':SIR,'t':0} 
        
        self.dispersion_kernel=np.array([[0.5, 1 , 0.5],
                                        [1  , -6, 1],
                                        [0.5, 1, 0.5]]) # SIR模型卷积核        
    
    def make_frame(self,t):
        '''返回每一步卷积的数据到VideoClip中'''
        while self.world['t']<self.hours_per_second*t:
            self.update(self.world) 
        if self.world['t']<6:
            print(self.world['SIR'][0])
        return self.world['SIR'][0]
    
    def update(self,world):        
        '''更新数组，即基于前一步卷积结果的每一步卷积'''
        disperse=self.dispersion(world['SIR'], self.dispersion_kernel)
        world['SIR']=disperse 
        world['t']+=dt  # 记录时间，用于循环终止条件
    
    def dispersion(self,SIR,dispersion_kernel):
        '''卷积扩散'''        
        return np.array([convolve(SIR[0],self.dispersion_kernel,mode='constant',cval=0.0)]) # 注意卷积核与待卷积数组的维度
    
    def execute_(self):       
        '''执行程序'''        
        
        self.animation=mpy.VideoClip(self.make_frame,duration=1) # duration=1
        self.animation.write_gif(self.save_path,self.fps)  


class SIR_spatialPropagating:
    '''
    funciton - SIR的空间传播模型
    
    Params:
        classi_array - 分类数据（.tif，或者其它图像类型），或者其它可用于成本计算的数据类型
        cost_mapping - 分类数据对应的成本值映射字典
        beta - beta值，确定S-->I的转换率
        gamma - gamma值，确定I-->R的转换率
        dispersion_rates - SIR三层栅格各自对应的卷积扩散率
        dt - 时间更新速度
        hours_per_second - 扩散时间长度/终止值(条件)
        duration - moviepy参数配置，持续时长
        fps - moviepy参数配置，每秒帧数
        SIR_gif_savePath - SIR空间传播计算结果.gif文件保存路径
        
    Examples
    --------
    # 成本栅格（数组）
    classi_array=mosaic_classi_array_rescaled    
    
    # 配置用地类型的成本值（空间阻力值）
    cost_H=250
    cost_M=125
    cost_L=50
    cost_Z=0
    cost_mapping={
                'never classified':(0,cost_Z),
                'unassigned':(1,cost_Z),
                'ground':(2,cost_M),
                'low vegetation':(3,cost_H),
                'medium vegetation':(4,cost_H),
                'high vegetation':(5,cost_H),
                'building':(6,cost_Z),
                'low point':(7,cost_Z),
                'reserved':(8,cost_M),
                'water':(9,cost_M),
                'rail':(10,cost_L),
                'road surface':(11,cost_L),
                'reserved':(12,cost_M),
                'wire-guard(shield)':(13,cost_M),
                'wire-conductor(phase)':(14,cost_M),
                'transimission':(15,cost_M),
                'wire-structure connector(insulator)':(16,cost_M),
                'bridge deck':(17,cost_L),
                'high noise':(18,cost_Z),
                'null':(9999,cost_Z)       
                }    

    # 参数配置
    start_pt=[418,640]  # [3724,3415]
    beta=0.3
    gamma=0.1
    dispersion_rates=[0, 0.07, 0.03]  # S层卷积扩散为0，I层卷积扩散为0.07，R层卷积扩散为0.03
    dt=1.0
    hours_per_second=30*24 # 7*24
    duration=12 #12
    fps=15 # 15
    SIR_gif_savePath=r"./imgs/SIR_sp.gif"
    
    SIR_sp=SIR_spatialPropagating(classi_array=classi_array,cost_mapping=cost_mapping,start_pt=start_pt,beta=beta,gamma=gamma,dispersion_rates=dispersion_rates,dt=dt,hours_per_second=hours_per_second,duration=duration,fps=fps,SIR_gif_savePath=SIR_gif_savePath)
    SIR_sp.execute()     
    '''
    def __init__(self,classi_array,cost_mapping,start_pt=[10,10],beta=0.3,gamma=0.1,dispersion_rates=[0, 0.07, 0.03],dt=1.0,hours_per_second=7*24,duration=12,fps=15,SIR_gif_savePath=r'./SIR_sp.gif'):
        # 将分类栅格，按照成本映射字典，转换为成本栅格(配置空间阻力)
        for idx,(identity,cost_value) in enumerate(cost_mapping.items()):
            classi_array[classi_array==cost_value[0]]=cost_value[1]
        self.mms=MinMaxScaler()
        normalize_costArray=self.mms.fit_transform(classi_array) # 标准化成本栅格

        # 配置SIR模型初始值，将S设置为空间阻力值
        SIR=np.zeros((3,normalize_costArray.shape[0], normalize_costArray.shape[1]),dtype=float)        
        SIR[0]=normalize_costArray
        
        # 配置SIR模型中I的初始值。1，可以从设置的1个或多个点开始；2，可以将森林部分直接设置为I有值，而其它部分保持0。
        # start_pt=int(0.7*normalize_costArray.shape[0]), int(0.2*normalize_costArray.shape[1])  #根据行列拾取点位置
        # print("起始点:",start_pt)
        start_pt=start_pt
        SIR[1,start_pt[0],start_pt[1]]=0.8  #配置起始点位置值

        # 配置转换系数，以及卷积核
        self.beta=beta # β值
        self.gamma=gamma # γ值
        self.dispersion_rates=dispersion_rates  # 扩散系数
        dispersion_kernelA=np.array([[0.5, 1 , 0.5],
                                     [1  , -6, 1],
                                     [0.5, 1, 0.5]])  # 卷积核_类型A    
        dispersion_kernelB=np.array([[0, 1 , 0],
                                     [1 ,1, 1],
                                     [0, 1, 0]])  # 卷积核_类型B  

        self.dispersion_kernel=dispersion_kernelA # 卷积核
        self.dt=dt  # 时间记录值，开始值
        self.hours_per_second=hours_per_second  # 终止值(条件) 
        self.world={'SIR':SIR,'t':0} # 建立字典，方便数据更新
        
        # moviepy配置
        self.duration=duration
        self.fps=fps        
                 
        # 保存路径
        self.SIR_gif_savePath=SIR_gif_savePath
    
    def deriv(self,SIR,beta,gamma):
        '''SIR模型'''
        S,I,R=SIR
        dSdt=-1*beta*I*S  
        dRdt=gamma*I
        dIdt=beta*I*S-gamma*I
        return np.array([dSdt, dIdt, dRdt])
    
    def dispersion(self,SIR,dispersion_kernel,dispersion_rates):
        '''卷积扩散'''
        
        return np.array([convolve(e,dispersion_kernel,cval=0)*r for (e,r) in zip(SIR,dispersion_rates)])
    
    def update(self,world):
        '''执行SIR模型和卷积，更新world字典'''
        deriv_infect=self.deriv(world['SIR'],self.beta,self.gamma)
        disperse=self.dispersion(world['SIR'], self.dispersion_kernel, self.dispersion_rates)
        world['SIR'] += self.dt*(deriv_infect+disperse)    
        world['t'] += self.dt
    
    def world_to_npimage(self,world):
        '''将模拟计算的值转换到[0,255]RGB色域空间'''
        coefs=np.array([2,20,25]).reshape((3,1,1))
        SIR_coefs=coefs*world['SIR']
        accentuated_world=255*SIR_coefs
        image=accentuated_world[::-1].swapaxes(0,2).swapaxes(0,1) # 调整数组格式为用于图片显示的（x,y,3）形式
        return np.minimum(255, image)
    
    def make_frame(self,t):
        '''返回每一步的SIR和卷积综合蔓延结果'''
        while self.world['t']<self.hours_per_second*t:
            self.update(self.world)     
        return self.world_to_npimage(self.world)    
    
    def execute(self):
        '''执行程序'''        
        animation=mpy.VideoClip(self.make_frame,duration=self.duration)  #12
        animation.write_gif(self.SIR_gif_savePath, fps=self.fps) #15

if __name__=="__main__":
    pass
    #------test SIR_deriv()  
    '''
    # 参数配置
    N=1000 # 总人口数
    I_0,R_0=1,0 # 初始化受感人群，及恢复人群的人口数
    S_0=N-I_0-R_0 # 有受感人群和恢复人群，计算得易感人群人口数
    beta,gamma=0.2,1./10 # 配置参数b(即beta)和k(即gamma)
    t=np.linspace(0,160,160) # 配置时间序列    
    y_0=S_0,I_0,R_0
    SIR_array=SIR_deriv(y_0,t,N,beta,gamma,plot=True)   
    '''
    
    