# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 17:59:54 2023

@author: richie bao
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 19:50:26 2023

@author: richie bao
"""
import dearpygui.dearpygui as dpg
import random
from src.chain_update import func_chain_update
from src.util import *

from fastai.data.all import *
from fastai.vision.all import *
import torch

from pathlib import Path
import os

def add_node_imgs_batch(user_data):
    # Create random ID and check that the ID does not exist yet for this node type  
    node_type = "!Node_imgs_batch"    
    random_id=unique_tag_randint(node_type)
        
    with dpg.node(tag=random_id + node_type,
                  parent="NodeEditor",
                  label="Load WGAN64 model",
                  pos=user_data):               
        with dpg.node_attribute(tag=str(random_id) + node_type + "_Input1"):
            dpg.add_input_text(tag=str(random_id) + node_type + "_Input1_value",
                                label="Images Folder",
                                width=100,
                                enabled=False,
                                readonly=True,
                                )
            
        with dpg.node_attribute(tag=str(random_id) + node_type + "_Input2"):    
            dpg.add_input_int(label="Batch size", 
                              tag=random_id+node_type + "_Input2_value",
                              width=100,
                              default_value=128)
            
        with dpg.node_attribute(tag=str(random_id) + node_type + "_Input3"):    
            dpg.add_input_int(label="Image size", 
                              tag=random_id+node_type + "_Input3_value",
                              width=100,
                              default_value=64)            
            
            
        with dpg.node_attribute(tag=str(random_id) + node_type + "_Cal" ):     
            dpg.add_button(label='alculate images batch',
                           tag=random_id+'!cal_imgs_batch',
                           callback=cal_imgs_batch,
                           user_data=[random_id,node_type],
                           arrow=True,
                           direction=1,)    
            
def cal_imgs_batch(sender, app_data, user_data):      
    random_id,node_type=user_data

    imgs_folder=Path(dpg.get_value(str(random_id) + node_type + "_Input1_value"))
    bs=dpg.get_value(random_id+node_type + "_Input2_value")
    size=dpg.get_value(random_id+node_type + "_Input3_value")    
    files=get_image_files(imgs_folder)
    
    def label_func(f): return 1
    # ngpu=1
    # device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    dls=ImageDataLoaders.from_name_func(imgs_folder, files, label_func,item_tfms=Resize(size),bs=bs)  # ,device=device
    x,_=dls.one_batch()
    # print(x.cpu())
    gvals['imgs_batch'+random_id]=x   
                    
    with dpg.node_attribute(tag=str(random_id) + node_type+ "_Output", attribute_type=dpg.mvNode_Attr_Output,parent=str(random_id) + node_type):
        dpg.add_text(tag=str(random_id) + node_type+ "_Output_value",
                      label="Image Batch ID",
                      default_value='imgs_batch'+random_id,
                      bullet=True)
        
if __name__=="__main__":
    imgs_folder=r'I:\data\NAIP4StyleGAN\patches_64'
    cal_imgs_batch(imgs_folder)
