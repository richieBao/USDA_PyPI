# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 08:49:11 2022

@author: richie bao
"""
import matplotlib

def generate_colors():    
    '''
    function - 生成颜色列表或者字典
    
    Returns:
        hex_colors_only - 16进制颜色值列表；list
        hex_colors_dic - 颜色名称：16进制颜色值；dict
        rgb_colors_dic - 颜色名称：(r,g,b)；dict
    '''
    hex_colors_dic={}
    rgb_colors_dic={}
    hex_colors_only=[]
    for name, hex in matplotlib.colors.cnames.items():
        hex_colors_only.append(hex)
        hex_colors_dic[name]=hex
        rgb_colors_dic[name]=matplotlib.colors.to_rgb(hex)
    return hex_colors_only,hex_colors_dic,rgb_colors_dic