# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 08:21:02 2023

@author: richie bao
"""
import dearpygui.dearpygui as dpg
from src.chain_update import func_chain_update
from src.util import *

from random import randint
import numpy as np
from PIL import Image

unique_tag_randint=general_util.unique_tag_randint
# img_tag_lst=[]

def add_node_image(user_data):
    # Create random ID and check that the ID does not exist yet for this node type
    random_id = randint(0, 50000)    
    node_type = "!Node_Img"
    
    while dpg.does_item_exist(str(random_id) + node_type):
        random_id = randint(0, 50000)
    
    rdn_file_dialog_id=unique_tag_randint('!file_dialog_id')
    with dpg.file_dialog(directory_selector=False, show=False,callback=callback, tag=rdn_file_dialog_id+"!file_dialog_id", width=700 ,height=400,user_data=[random_id,node_type]): 
        dpg.add_file_extension(".*")
        dpg.add_file_extension("", color=(150, 255, 150, 255))
        dpg.add_file_extension("Source files (*.jpg *.png *.bmp){.psd,.gif,.hdr}", color=(0, 255, 255, 255))
        dpg.add_file_extension(".pic", color=(255, 0, 255, 255), custom_text="[header]")
        dpg.add_file_extension(".ppm", color=(0, 255, 0, 255), custom_text="[Python]")       
            
    with dpg.node(tag=str(random_id) + "!Node_img",
                  parent="NodeEditor",
                  label="Input image",
                  pos=user_data):    
        with dpg.node_attribute(tag=str(random_id) + "_img_selector"):
            dpg.add_button(tag=str(random_id)+'!b_img_1',label="File Selector", callback=lambda: dpg.show_item(rdn_file_dialog_id+"!file_dialog_id"))
            
def callback(sender, app_data, user_data):  
    random_id,node_type=user_data
    img_tag_lst=[str(random_id)+'!img_path',str(random_id)+"!texture_tag",str(random_id)+'!img_show',str(random_id) + node_type+ "_Output_value",str(random_id) + node_type+ "_Output"]    
        
    for i in img_tag_lst:        
        if dpg.does_item_exist(i):
            dpg.delete_item(i)  
    
    parent_tag=str(random_id) + "_img_selector"
    img_path=list(app_data['selections'].values())[0]
    dpg.add_text(img_path,tag=str(random_id)+'!img_path',parent=parent_tag) 
    height,width,channels,data,original_im_array=resize_img(img_path,200)  
    gvals['imarray_'+str(random_id)]=original_im_array

    with dpg.texture_registry(show=False):
        dpg.add_static_texture(width=width, height=height, default_value=data, tag=str(random_id)+"!texture_tag")

    dpg.add_image(str(random_id)+"!texture_tag",parent=parent_tag,tag=str(random_id)+'!img_show')    

    with dpg.node_attribute(tag=str(random_id) + node_type+ "_Output", attribute_type=dpg.mvNode_Attr_Output,parent=str(random_id) + "!Node_img"):
        dpg.add_text(tag=str(random_id) + node_type+ "_Output_value",
                     label="Image array key",
                     default_value='imarray_'+str(random_id),
                     bullet=True)
    
def resize_img(img_path,basewidth):
    im=Image.open(img_path)
    wpercent=(basewidth/float(im.size[0]))
    hsize=int((float(im.size[1])*float(wpercent)))    
    scaled_im=im.resize((basewidth,hsize), Image.Resampling.LANCZOS)
    scaled_im=scaled_im.convert('RGBA')
    # scaled_im.show()
    im_array=np.array(scaled_im)
    
    width, height, channels=im_array.shape
    im_array_flatten=im_array.flatten()/255
    # print(im_array.shape)
    return  width, height,channels, im_array_flatten.tolist(),np.array(im)
    
        
if __name__=="__main__":
    img_path=r'C:\Users\richi\omen_richiebao\omen_temp\2.jpg'
    channels,width, height,data,original_im_array=resize_img(img_path,200)
    
    