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
from fastai.callback.all import *
from fastai.vision.all import *
from fastai.vision.gan import *

from pathlib import Path
import os
from PIL import Image
import torchvision.utils as vutils
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F
plt.rcParams["savefig.bbox"] = 'tight'

def add_node_wgan_64(user_data):
    # Create random ID and check that the ID does not exist yet for this node type  
    node_type = "!Node_Model"    
    random_id=unique_tag_randint(node_type)
    
    # rdn_file_dialog_id=unique_tag_randint('!file_dialog_id')
    with dpg.file_dialog(directory_selector=False, show=False,callback=callback, tag=random_id+"!file_dialog_id", width=700 ,height=400,user_data=[random_id,node_type]): 
        dpg.add_file_extension("Source files (*.pth *.pkl *.ckp){.pth,.pkl,.ckp}",color=(255, 0, 255, 255))        
    dpg.add_file_dialog(directory_selector=True, show=False,callback=callback, tag=random_id+"!directory_dialog_id", width=700 ,height=400,user_data=[random_id,node_type])
            
    with dpg.node(tag=random_id + "!Node_img",
                  parent="NodeEditor",
                  label="Load model",
                  pos=user_data):    
        with dpg.node_attribute(tag=random_id + "_model_selector"):
            dpg.add_button(tag=random_id+'!b_model',label="Model Selector", callback=lambda: dpg.show_item(random_id+"!file_dialog_id"))
            dpg.add_button(tag=random_id+'!b_img_folder',label="Imgs Folder Selector", callback=lambda: dpg.show_item(random_id+"!directory_dialog_id"))
            
def callback(sender, app_data, user_data):  
    random_id,node_type=user_data
    # img_tag_lst=[str(random_id)+'!img_path',str(random_id)+"!texture_tag",str(random_id)+'!img_show',str(random_id) + node_type+ "_Output_value",str(random_id) + node_type+ "_Output"]    
        
    # for i in img_tag_lst:        
    #     if dpg.does_item_exist(i):
    #         dpg.delete_item(i)  
    
    model_path,imge_folder=None,None    
    layers,img_info=None,None 
    
    parent_tag=str(random_id)+ "_model_selector"
    if sender.split('!')[1]=='file_dialog_id':
        model_path=list(app_data['selections'].values())[0]
        dpg.add_text(model_path,tag=random_id+'!text_model_path',parent=parent_tag)        
    elif sender.split('!')[1]=='directory_dialog_id':        
        imge_folder=app_data['file_path_name']
        dpg.add_text(imge_folder,tag=random_id+'!text_imgs_folder',parent=parent_tag) 
    
    if model_path is not None and imge_folder is not None:
        layers,img_info=(model_path.split('.')[0],imge_folder)            
    
    if layers is not None and img_info is not None:
        imgs_texture_tag=_create_static_textures(img_info,random_id)
        print(imgs_texture_tag)
        
    print('-'*50)
    # with dpg.plot(label="Image Plot", height=400, width=-1,parent=parent_tag):
    #     dpg.add_plot_legend()
    #     dpg.add_plot_axis(dpg.mvXAxis, label="x axis")
    #     with dpg.plot_axis(dpg.mvYAxis, label="y axis"):
            
            
    #         dpg.add_image_series(dpg.mvFontAtlas, [300, 300], [400, 400], label="font atlas")
    #         dpg.add_image_series("__demo_static_texture_1", [0, 0], [100, 100], label="static 1")
    #         dpg.add_image_series("__demo_static_texture_2", [150, 150], [200, 200], label="static 2")
    #         dpg.add_image_series("__demo_static_texture_3", [200, -150], [300, -50], label="static 3")
    #         dpg.add_image_series("__demo_dynamic_texture_1", [-200, 100], [-100, 200], label="dynamic 1")
    #         dpg.add_image_series("__demo_dynamic_texture_2", [-200, -100], [-150, -50], label="dynamic 2")

    
    
    
def gan_model(model_path,imgs_folder):
    size=64
    bs=128
    
    dblock=DataBlock(
        blocks=(TransformBlock, ImageBlock),
        get_x=generate_noise,
        get_items=get_image_files,
        splitter=IndexSplitter([]),
        item_tfms=Resize(size, method=ResizeMethod.Crop), 
        batch_tfms=Normalize.from_stats(torch.tensor([0.5,0.5,0.5]), torch.tensor([0.5,0.5,0.5])))    
    
    dls=dblock.dataloaders(imgs_folder, path=imgs_folder, bs=bs)
    generator=basic_generator(64, n_channels=3, n_extra_layers=1)
    critic=basic_critic(64, n_channels=3, n_extra_layers=1, act_cls=partial(nn.LeakyReLU, negative_slope=0.2))    
    
    generator=basic_generator(64, n_channels=3, n_extra_layers=1)
    critic=basic_critic(64, n_channels=3, n_extra_layers=1, act_cls=partial(nn.LeakyReLU, negative_slope=0.2))
    
    learn=GANLearner.wgan(dls, generator, critic, opt_func=RMSProp,
                          lr=1e-5, 
                          clip=0.01,
                          switcher=FixedGANSwitcher(n_crit=5, n_gen=1),
                          switch_eval=False
                          )    
    
    learn.recorder.train_metrics=True
    learn.recorder.valid_metrics=False    
    learn.load(model_path,with_opt=True)    
      
    nz=100    
    netG=learn.model.generator 
    get_x=partial(generate_noise,size=nz)
    fixed_noise=get_x(nz)[None,:]
    # fake=netG(fixed_noise).detach().cpu()
    # fake_array=np.transpose(fake[0],(1,2,0))
    
    layers=[module for module in netG.modules() if not isinstance(module, nn.Sequential)]
    imgs_dict={}
    for idx,layer in enumerate(layers):
        if idx==0:
            fake=layer(fixed_noise)
            imgs_dict[idx]=fake.detach().cpu()[0]
        else:
            fake=layer(fake)
            imgs_dict[idx]=fake.detach().cpu()[0]
            
    img_info=sorting_layer_imgs4dpg(imgs_dict)   
    
    
    return layers,img_info   

def sorting_layer_imgs4dpg(layer_imgs):
    imgs_T_dict={}
    ch_size_dict={100:[100,10],512:[100,10],256:[100,10],128:[100,10],64:[64,8],3:[3,64]}
    
    img_info={}
    for k,v in layer_imgs.items():        
        ch,size=ch_size_dict[v.shape[0]]   
        if ch==3:
            img=v[:ch][None,:,:,:]            
        else:
            img=v[:ch][:,None,:,:]  
        grid=vutils.make_grid(img, padding=1, normalize=True)
        # show(grid)
        grid_pil=F.to_pil_image(grid)
        grid_rgba=grid_pil.convert('RGBA')
        grid_rgba_array=np.array(grid_rgba)
        width, height, channels=grid_rgba_array.shape
        im_flatten=grid_rgba_array.flatten()/255
        
        img_info[k]=[width, height, channels,im_flatten]
    
    return img_info

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        
def _create_static_textures(img_info,random_id):
    print('#'*50)
    parent_tag=random_id+"_wgan_texture_container"
    dpg.add_texture_registry(label="wgan_texture_container", tag=parent_tag)
    imgs_texture_tag=[]
    for k,v in img_info.items():
        width, height, channels,im_flatten=v
        dpg.add_static_texture(width,height,im_flatten, parent=parent_tag, tag=random_id+f"layer_{k}", label=f"layer_{k}")
        imgs_texture_tag.append(random_id+f"layer_{k}")
        
    return imgs_texture_tag
    
if __name__=="__main__":
    model_path=r'C:\Users\richi\omen_richiebao\omen_github\USDA_PyPI\src\usda\tools\DL_layers_visualizer\naip_wgan_c_learn'
    imgs_folder=r'I:\data\NAIP4StyleGAN\patches_64'
    gan_model(model_path,imgs_folder)