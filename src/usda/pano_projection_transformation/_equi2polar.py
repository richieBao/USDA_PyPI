# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 17:09:51 2023

@author: richie bao
"""
import glob,os
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial

if __package__:
    from ._equi2polar_pool import equi2polar_pool
else:
    from _equi2polar_pool import equi2polar_pool
    
def equi2polar(pano_path,output_shape,save_path,little_planet='little_planet_1',cpu_num=8):
    img_fns=glob.glob(os.path.join(pano_path,'*.jpg'))

    args=partial(equi2polar_pool, args=[output_shape,little_planet,save_path])    
    with Pool(cpu_num) as p:
        p.map(args, tqdm(img_fns))      


if __name__ == '__main__':  
    import usda.utils as usda_utils
    __C=usda_utils.AttrDict()
    args=__C
    __C.pano_path=r'G:\data\pano_dongxistreet\pano_projection_transforms\img_seg_redefined_color'
    __C.output_shape=(1024,1024)
    __C.little_planet='little_planet_1'
    __C.polar_seg_img_dir=r'G:\data\pano_dongxistreet\polar_seg_img'
        
    pano_path=args.pano_path
    output_shape=args.output_shape
    little_planet=args.little_planet
    polar_save_path=args.polar_seg_img_dir    
    
    equi2polar(pano_path,output_shape,polar_save_path,little_planet='little_planet_1')