# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 14:30:37 2022

@author: richie bao
"""
import matplotlib.pyplot as plt
import math,os
import numpy as np
from rio_tiler.io import COGReader
from skimage import exposure
from rasterio.plot import show 

def Sentinel2_bandsComposite_show(RGB_bands,zoom=10,tilesize=512,figsize=(10,10)):
    '''
    function - Sentinel-2波段合成显示。需要deg2num(lat_deg, lon_deg, zoom)和centroid(bounds)函数
    
    Params:
        RGB_bands - 波段文件路径名字典，例如{"R":path_R,"G":path_G,"B":path_B}；dict
        zoom - zoom level缩放级别。The defalut is 10；int
        tilesize - 瓦片大小。The default is 512；int
        figsize- 打印图表大小。The default is (10,10)；tuple
        
    Returns:
        None
    '''   
    
    B_band=RGB_bands["B"]
    G_band=RGB_bands["G"]
    R_band=RGB_bands["R"]
    
    def band_reader(band):
        with COGReader(band) as image:
            bounds=image.geographic_bounds
            print('影像边界坐标：',bounds)
            x, y=deg2num(*centroid(bounds), zoom)
            print("影像中心点瓦片索引：",x,y)
            img=image.tile(x, y, zoom, tilesize=tilesize) 
            return img.data
        
    tile_RGB_list=[np.squeeze(band_reader(band)) for band in RGB_bands.values()]
    tile_RGB_array=np.array(tile_RGB_list).transpose(1,2,0)
    p2, p98=np.percentile(tile_RGB_array, (2,98))
    image=exposure.rescale_intensity(tile_RGB_array, in_range=(p2, p98)) / 65535
    
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.axis("off")
    plt.show()
    