# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 14:47:19 2023

@author: richie bao
"""
from PIL import Image
import torchvision.transforms as T
from yacs.config import CfgNode as CN
import torch

if __package__:
    from .data import get_transform
    from .models import define_G   
    from usda.migrated_project.stylegan import adjust_dynamic_range
else:
    from data import get_transform
    from models import define_G
    from usda.migrated_project.stylegan import adjust_dynamic_range

def A2B_generator(pretrained_model_fn,AorB_fn,cfg=None):
    opt=CN()
    opt.dataset = CN()
    opt.dataset.preprocess=['resize','crop']
    opt.dataset.load_size=512
    opt.dataset.crop_size=512   
    opt.dataset.no_flip=True
    
    opt.model = CN()
    opt.model.input_nc=3
    opt.model.output_nc=3
    opt.model.ngf=64
    opt.model.netG='unet_256'    

    try:
        opt.dataset.update(cfg.dataset)
        opt.model.update(cfg.model)
    except:
        pass        
    
    # print(opt)
    state_dict=torch.load(pretrained_model_fn)
    G_net=define_G(input_nc=opt.model.input_nc,
                          output_nc=opt.model.output_nc, 
                          ngf=opt.model.ngf, 
                          netG=opt.model.netG)
    G_net.load_state_dict(state_dict)    
    
    AorB=Image.open(AorB_fn).convert('RGB')
    rfms=get_transform(opt)     
    AorB_trfm=rfms(AorB)[None,:]
    
    g_BorA=G_net(AorB_trfm)
    g_BorA_adj=adjust_dynamic_range(g_BorA)  
    
    trf_tensor2im=T.ToPILImage()
    BorA=trf_tensor2im(g_BorA_adj[0])
    
    return AorB,BorA 

if __name__=="__main__":
    pretrained_model_fn=r'I:\model_ckpts\pix2pix\pix2pix4LC2IMG\latest_net_G.pth'
    img_fn=r'I:\model_ckpts\pix2pix\pix2pix4LC2IMG\web\images\epoch001_real_A.png'
    cfg=CN()
    cfg.dataset=CN()
    cfg.dataset.no_flip=True
    
    AorB,BorA=A2B_generator(pretrained_model_fn,img_fn,cfg)