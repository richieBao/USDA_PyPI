# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 12:04:13 2023

@author: richie bao
"""
from PIL import Image
import cv2
import skimage.exposure
import numpy as np
from numpy.random import default_rng
from PIL import Image

def imgs_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def imgs_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def random_shape_onImage(img,thresh1=130,thresh2=255):
    # define random seed to change the pattern
    #seedval=75
    rng=default_rng() # seed=seedval
    # create random noise image
    height, width = img.shape[:2]
    noise=rng.integers(0, 255, (height,width), np.uint8, True)
    # blur the noise image to control the size
    blur=cv2.GaussianBlur(noise, (0,0), sigmaX=15, sigmaY=15, borderType = cv2.BORDER_DEFAULT)
    # stretch the blurred image to full dynamic range
    stretch=skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0,255)).astype(np.uint8)
    # threshold stretched image to control the size
    thresh=cv2.threshold(stretch, thresh1, thresh2, cv2.THRESH_BINARY)[1]
    # apply morphology open and close to smooth out and make 3 channels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,  (9,9))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.merge([mask,mask,mask])
    # add mask to input
    result = cv2.add(img, mask)    
    PIL_img = Image.fromarray(np.uint8(result)).convert('RGB')
    
    return result,PIL_img

def binarize_noise_image(img_fn,channel=0,threshold_binary=0.9,direction='gt',threshold_noise=0.1,resize=None):
    
    img=Image.open(img_fn)
    if resize:
        img_original=img.resize(resize)
    else:
        img_original=img        
    
    img_original_mono=np.array(img_original)[:,:,channel]/255
    
    if direction=='gt':
        img_original_mono=np.where(img_original_mono>threshold_binary,1,-1)
    elif direction=='lt':
        img_original_mono=np.where(img_original_mono<threshold_binary,1,-1)
        
    img_noise=img_original_mono.copy()
    n=img_original_mono.shape[0]
    noise=np.random.rand(n,n)
    ind=np.where(noise<threshold_noise)
    img_noise[ind]=-img_noise[ind]
    
    return img_original_mono,img_noise


if __name__=="__main__":
    import glob, os
    naip_512_path=r'I:\data\naip_lc4pix2pix\imgs'
    img_fns=glob.glob(naip_512_path+"/*.jpg")
    img=cv2.imread(img_fns[0])
    _,img=random_shape_onImage(img)
