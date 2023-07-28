# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 10:56:03 2023

@author: richie bao
"""
import glob,os
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial

if __package__:
    from ._equi2cube_pool import labels_equi2cube_pool
else:
    from _equi2cube_pool import labels_equi2cube_pool
    
def equi2cube(label_seg_fns,equi2cub_dir,face_size,cpu_num=8):
    save_path_dict={"label_seg_cube_root":os.path.join(equi2cub_dir,'cube_label_seg'),
                    "img_seg_redefined_color_root":os.path.join(equi2cub_dir,'img_seg_redefined_color'),
                    "img_seg_cube_root":os.path.join(equi2cub_dir,'cube_img_seg')}   
    
    args=partial(labels_equi2cube_pool, args=[save_path_dict,face_size])
    with Pool(cpu_num) as p:
        p.map(args, tqdm(label_seg_fns))         

if __name__ == '__main__':  
    import usda.utils as usda_utils
    __C=usda_utils.AttrDict()
    args=__C
    __C.pano_path=r'G:\data\pano_dongxistreet\images_valid'
    __C.label_seg_path=r'G:\data\pano_dongxistreet\pano_seg\seg_label'   
    __C.face_size=1000
    __C.equi2cub_dir=r'G:\data\pano_dongxistreet\pano_projection_transforms'
    

    label_seg_path=args.label_seg_path
    label_seg_fns=glob.glob(os.path.join(label_seg_path,'*.pkl'))
    equi2cube_dir=args.equi2cub_dir
    face_size=args.face_size
    
    equi2cube(label_seg_fns,equi2cub_dir,face_size,cpu_num=8)
    
    