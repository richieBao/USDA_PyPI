# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:08:33 2023

@author: richie bao
migrated from: PASS:https://github.com/elnino9ykl/PASS
"""
if __package__:
    from ._erfnet_pspnet import Net
    from ._dataset import cityscapes
    from ._transform import Relabel, ToLabel, Colorize
else:
    from _erfnet_pspnet import Net
    from _dataset import cityscapes
    from _transform import Relabel, ToLabel, Colorize
    
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
from tqdm import tqdm
import os
from pathlib import Path
import pickle

    
NUM_CHANNELS = 3
NUM_CLASSES = 28

image_transform = ToPILImage()
input_transform_cityscapes = Compose([
    Resize((512,1024*1),Image.BILINEAR),
    ToTensor(),
])

def load_state_dict(model, state_dict):  #custom function to load model when not all dict elements
    own_state = model.state_dict()
    
    # for a in own_state.keys():
    #     print(a)
    # for a in state_dict.keys():
    #     print(a)
    # print('-----------')
    
    for name, param in state_dict.items():
        if name not in own_state:
             continue
        own_state[name].copy_(param)
    
    return model

def pass_seg(weights_fn,pano_dir,save_dir,cpu=False,num_workers=10,batch_size=1):
    model=Net(NUM_CLASSES)
    model=torch.nn.DataParallel(model)
    
    if (not cpu):
        model=model.cuda()
        
    model=load_state_dict(model, torch.load(weights_fn))
    print ("Model and weights LOADED successfully")
    model.eval()
    
    if(not pano_dir):
        print ("Error: datadir could not be loaded")    
    
    loader=DataLoader(cityscapes(pano_dir, input_transform_cityscapes, subset="val"),num_workers=num_workers, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for step, (images, filename) in enumerate(tqdm(loader)):
            images_A=images.cuda()
            outputsA=model(images_A)
            outputs=outputsA
            
            save_img_dir=os.path.join(save_dir,'seg_img')
            save_label_dir=os.path.join(save_dir,'seg_label')
            os.makedirs(save_img_dir, exist_ok=True)
            os.makedirs(save_label_dir, exist_ok=True)            
            
            label=outputs[0].cpu().max(0)[1].data.byte()
            with open(os.path.join(save_label_dir,'{}.pkl'.format(Path(filename[0]).stem)),'wb') as f:
                pickle.dump(label,f)    
                
            label_color=Colorize()(label.unsqueeze(0))

            label_save=ToPILImage()(label_color)
            filenameSave=os.path.join(save_img_dir,os.path.basename(filename[0]))
            label_save.save(filenameSave) 
            # print (step, filenameSave)
            # break

if __name__=="__main__":
    weights_fn=r'C:\Users\richie\omen_richiebao\omen_github\USDA_special_study\models\erfpspnet.pth'
    pano_dir=r'G:\data\pano_dongxistreet\images_valid'
    save_dir=r'G:\data\pano_dongxistreet\pano_seg'
    pass_seg(weights_fn,pano_dir,save_dir)
    
    # fn=r'G:\data\pano_dongxistreet\pano_seg\seg_label\dongxistreet_0.pkl'
    # with open(fn,'rb') as f:
    #     lab=pickle.load(f)
    