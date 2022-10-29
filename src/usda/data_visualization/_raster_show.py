# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 08:11:31 2022

@author: richie bao
"""
import matplotlib.pyplot as plt
from rasterio.plot import plotting_extent
import earthpy.plot as ep

def bands_show(img_stack_list,band_num):
    '''
    function - 指定波段，同时显示多个遥感影像
    
    Params:
        img_stack_list - 影像列表；list(array)
        band_num - 显示的层列表；list(int)
    '''   
    
    def variable_name(var):
        '''
        function - 将变量名转换为字符串
        
        Parasm:
            var - 变量名
        '''
        return [tpl[0] for tpl in filter(lambda x: var is x[1], globals().items())][0]

    plt.rcParams.update({'font.size': 12})
    img_num=len(img_stack_list)
    fig, axs = plt.subplots(1,img_num,figsize=(12*img_num, 12))
    i=0
    for img in img_stack_list:
        ep.plot_rgb(
                    img,
                    rgb=band_num,
                    stretch=True,
                    str_clip=0.5,
                    title="%s"%variable_name(img),
                    ax=axs[i]
                )
        i+=1
    plt.show()
    
    
if __name__=="__main__":
    pass

