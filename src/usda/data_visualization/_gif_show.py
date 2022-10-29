# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 19:18:23 2022

@author: richie bao
"""
from PIL import Image, ImageSequence
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML 

def animated_gif_show(gif_fp,figsize=(8,8)):
    '''
    function - 读入.gif，并动态显示
    
    Params:
        gif_fp - GIF文件路径；string
        figsize - 图表大小，The default is (8,8)；tuple
        
    Returns:
        HTML
    '''
      
    gif=Image.open(gif_fp,'r')    
    frames=[np.array(frame.getdata(),dtype=np.uint8).reshape(gif.size[0],gif.size[1],-1) for frame in ImageSequence.Iterator(gif)]

    fig=plt.figure(figsize=figsize)
    imgs=[(plt.imshow(img,animated=True),) for img in frames]
    anim=animation.ArtistAnimation(fig,imgs,interval=300,repeat_delay=3000,blit=True)  
    
    return HTML(anim.to_html5_video())

