# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 17:14:27 2023

@author: richie bao
ref: http://www.richwareham.com/little-planet-projection/
"""
import glob,os 
from tqdm import tqdm
from PIL import Image,ImageOps
import numpy as np
from skimage.transform import warp
from PIL import Image
from pathlib import Path
 

# import sys
# sys.path.append('..')

def output_coord_to_r_theta(coords):
    """Convert co-ordinates in the output image to r, theta co-ordinates.
    The r co-ordinate is scaled to range from from 0 to 1. The theta
    co-ordinate is scaled to range from 0 to 1.
    
    A Nx2 array is returned with r being the first column and theta being
    the second.
    """    
    # Calculate x- and y-co-ordinate offsets from the centre:
    x_offset = coords[:,0] - (output_shape[1]/2)
    y_offset = coords[:,1] - (output_shape[0]/2)
    
    # Calculate r and theta in pixels and radians:
    r = np.sqrt(x_offset ** 2 + y_offset ** 2)
    theta = np.arctan2(y_offset, x_offset)
    
    # The maximum value r can take is the diagonal corner:
    max_x_offset, max_y_offset = output_shape[1]/2, output_shape[0]/2
    max_r = np.sqrt(max_x_offset ** 2 + max_y_offset ** 2)
    
    # Scale r to lie between 0 and 1
    r = r / max_r
    
    # arctan2 returns an angle in radians between -pi and +pi. Re-scale
    # it to lie between 0 and 1
    theta = (theta + np.pi) / (2*np.pi)
    
    # Stack r and theta together into one array. Note that r and theta are initially
    # 1-d or "1xN" arrays and so we vertically stack them and then transpose
    # to get the desired output.
    return np.vstack((r, theta)).T

def r_theta_to_input_coords(r_theta):
    """Convert a Nx2 array of r, theta co-ordinates into the corresponding
    co-ordinates in the input image.
    
    Return a Nx2 array of input image co-ordinates.
    
    """
    # Extract r and theta from input
    r, theta = r_theta[:,0], r_theta[:,1]
    
    # Theta wraps at the side of the image. That is to say that theta=1.1
    # is equivalent to theta=0.1 => just extract the fractional part of
    # theta
    theta = theta - np.floor(theta)
    
    # Calculate the maximum x- and y-co-ordinates
    max_x, max_y = input_shape[1]-1, input_shape[0]-1
    
    # Calculate x co-ordinates from theta
    xs = theta * max_x
    
    # Calculate y co-ordinates from r noting that r=0 means maximum y
    # and r=1 means minimum y
    ys = (1-r) * max_y
    
    # Return the x- and y-co-ordinates stacked into a single Nx2 array
    return np.hstack((xs, ys))

def little_planet_1(coords):
    """Chain our two mapping functions together."""
    r_theta = output_coord_to_r_theta(coords)
    input_coords = r_theta_to_input_coords(r_theta)
    return input_coords

def little_planet_2(coords):
    """Chain our two mapping functions together with modified r."""
    r_theta = output_coord_to_r_theta(coords)
    # Take square root of r
    r_theta[:,0] = np.sqrt(r_theta[:,0])
    input_coords = r_theta_to_input_coords(r_theta)
    return input_coords

def little_planet_3(coords):
    """Chain our two mapping functions together with modified r
    and shifted theta.
    
    """
    r_theta = output_coord_to_r_theta(coords)
    
    # Take square root of r
    r_theta[:,0] = np.sqrt(r_theta[:,0])
    
    # Shift theta
    r_theta[:,1] += 0.1
    
    input_coords = r_theta_to_input_coords(r_theta)
    return input_coords

def little_planet_4(coords):
    """Chain our two mapping functions together with modified and
    scaled r and shifted theta.
    
    """
    r_theta = output_coord_to_r_theta(coords)
    
    # Scale r down a little to zoom in
    r_theta[:,0] *= 0.75
    
    # Take square root of r
    r_theta[:,0] = np.sqrt(r_theta[:,0])
    
    # Shift theta
    r_theta[:,1] += 0.1
    
    input_coords = r_theta_to_input_coords(r_theta)
    return input_coords

def equi2polar(imgs_root,output_shape,little_planet,save_path):
    '''
    转换等量矩形全景图为极坐标格式全景图

    Parameters
    ----------
    imgs_root : string
        全景图根目录.
    output_shape : tuple
        极坐标格式全景图的图像大小（width,height）.
    little_planet : function
        极坐标格式全景图可选类型：little_planet_1，little_planet_2，little_planet_3，little_planet_4.
    save_path : string
        图像保存根目录.

    Returns
    -------
    None.

    '''    
    img_fns=glob.glob(os.path.join(imgs_root,'*.jpg'))
    for fn in tqdm(img_fns):        
        pano=np.asarray(ImageOps.flip(Image.open(fn)))
        global input_shape
        input_shape=pano.shape      
        pano_warp=warp(pano, little_planet, output_shape=output_shape)
        # The image is a NxMx3 array of floating point values from 0 to 1. Convert this to
        # bytes from 0 to 255 for saving the image:
        pano_warp=(255 * pano_warp).astype(np.uint8)       
        im=Image.fromarray(pano_warp)
        # im_save_fn=os.path.join('./processed data/polar_img','{}.jpg'.format(Path(fn).stem))
        im_save_fn=os.path.join(save_path,'{}.jpg'.format(Path(fn).stem))
        im.save(im_save_fn)     
        break  

def equi2polar_pool(fn,args): 
    '''
    equi2polar(imgs_root,output_shape,little_planet,save_path)函数的多进程版。转换等量矩形全景图为极坐标格式全景图

    Parameters
    ----------
    fn : string
        单个图像路径.
    args : list
        包括：output_shape,little_planet,save_path 等3个参数.

    Returns
    -------
    None.

    '''    
    global output_shape
    output_shape,little_planet,save_path=args
       
    pano=np.asarray(ImageOps.flip(Image.open(fn)))
    global input_shape
    input_shape = pano.shape     
    pano_warp=warp(pano, eval(little_planet), output_shape=output_shape)
    pano_warp = (255 * pano_warp).astype(np.uint8)    
    # print(pano_warp.shape)
    im=Image.fromarray(pano_warp) 
    # print('+++')
    im_save_fn=os.path.join(save_path,'{}.jpg'.format(Path(fn).stem))
    im.save(im_save_fn)    

if __name__=="__main__":
    import os
    from database import cfg_load_yaml
    cfg=cfg_load_yaml('../config.yml')  
    
    parent_path=os.path.dirname(os.getcwd())
    imgs_root=os.path.join(parent_path,cfg["equi2cube"]["region"]["img_seg_redefined_color_root"]) 
    output_shape=eval(cfg["equi2polar"]["output_shape"])
    little_planet=eval(cfg["equi2polar"]["little_planet"])
    save_path=os.path.join(parent_path,cfg["equi2polar"]["region"]["polar_seg_root"])
    equi2polar(imgs_root,output_shape,little_planet,save_path)
      
