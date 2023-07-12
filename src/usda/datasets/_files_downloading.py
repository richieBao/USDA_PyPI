# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 14:43:38 2023

@author: richie bao
"""
import os
import urllib.request
from urllib.error import HTTPError

from torchvision import transforms
from torchvision.datasets import CIFAR10
import pytorch_lightning as pl

import torch
import torch.utils.data as data

import requests
import geopandas as gpd
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import shapely
from tqdm import tqdm

def files_downloading(base_url,files_name,save_dir):
    # Create checkpoint path if it doesn't exist yet
    os.makedirs(save_dir, exist_ok=True)
    
    # For each file, check whether it already exists. If not, try downloading it.
    for file_name in files_name:
        file_path = os.path.join(save_dir, file_name.split("/",1)[1])
        if "/" in file_name.split("/",1)[1]:
            os.makedirs(file_path.rsplit("/",1)[0], exist_ok=True)
        if not os.path.isfile(file_path):
            file_url = base_url + file_name
            print(f"Downloading {file_url}...")
            try:
                urllib.request.urlretrieve(file_url, file_path)
            except HTTPError as e:
                print("Something went wrong.", e)    

def cifar10_downloading2fixedParams_loader(dataset_path,seed=42):   
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
                                         ])
    # For training, we add some augmentation. Networks are too powerful and would overfit.
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomResizedCrop((32,32),scale=(0.8,1.0),ratio=(0.9,1.1)),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
                                         ])
    # Loading the training dataset. We need to split it into a training and validation part
    # We need to do a little trick because the validation set should not use the augmentation.
    train_dataset = CIFAR10(root=dataset_path, train=True, transform=train_transform, download=True)
    val_dataset = CIFAR10(root=dataset_path, train=True, transform=test_transform, download=True)
    pl.seed_everything(seed)
    train_set, _ = torch.utils.data.random_split(train_dataset, [45000, 5000])
    pl.seed_everything(seed)
    _, val_set = torch.utils.data.random_split(val_dataset, [45000, 5000])
    
    # Loading the test set
    test_set = CIFAR10(root=dataset_path, train=False, transform=test_transform, download=True)
    
    # We define a set of data loaders that we can use for various purposes later.
    train_loader = data.DataLoader(train_set, batch_size=128, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    val_loader = data.DataLoader(val_set, batch_size=128, shuffle=False, drop_last=False, num_workers=4)
    test_loader = data.DataLoader(test_set, batch_size=128, shuffle=False, drop_last=False, num_workers=4)   
    
    return train_set,val_set,test_set,train_loader,val_loader,test_loader 
    
def esa_worldcover_2020_grid_downloading(show=False,figsize=(20,10)):
    s3_url_prefix = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
    url=f"{s3_url_prefix}/v100/2020/esa_worldcover_2020_grid.geojson"
    #print(url) 
    response=requests.get(url)
    data=response.json()
    wc_grid=gpd.GeoDataFrame.from_features(data)
    
    if show:
        ranodm_cmap=mpl.colors.ListedColormap (np.random.rand(len(wc_grid),3))
        
        fig, ax=plt.subplots(figsize=figsize)
        wc_grid.plot(column='ll_tile',cmap=ranodm_cmap,ax=ax)
        fig.patch.set_visible(False)
        ax.axis('off')
        plt.show()    
        
        return wc_grid,ranodm_cmap
    else:
        return wc_grid
    
def esa_worldcover_downloading(boundary,save_root,y='2020'):
    wc_grid=esa_worldcover_2020_grid_downloading()
    bounds_geometry=shapely.Polygon.from_bounds(*boundary)
    
    # get grid tiles intersecting AOI
    tiles=wc_grid[wc_grid.intersects(bounds_geometry)]    
    
    url_lst=[]
    fns=[]
    s3_url_prefix = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
    for tile in tqdm(tiles.ll_tile):
        if y=='2020':
            url = f"{s3_url_prefix}/v100/2020/map/ESA_WorldCover_10m_2020_v100_{tile}_Map.tif"
        elif y=='2021':
            url = f"{s3_url_prefix}/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
        
        url_lst.append(url)
        r = requests.get(url, allow_redirects=True)
        out_fn = os.path.join(save_root,f"ESA_WorldCover_10m_2020_v100_{tile}_Map.tif")
        fns.append(out_fn)
        with open(out_fn, 'wb') as f:
            f.write(r.content)        
            
    return url_lst,fns,tiles
            