"""
-------------------------------------------------
   File Name:    generate_samples.py
   Date:         2019/10/27
   Description:  Generate single image samples from a particular depth of a model
                 Modified from: https://github.com/akanimax/pro_gan_pytorch
-------------------------------------------------
"""

import os
import numpy as np
from tqdm import tqdm

import torch
from torchvision.utils import save_image

from .models._GAN import Generator
from ._config import cfg as opt
from yacs.config import CfgNode as CN

def adjust_dynamic_range(data, drange_in=(-1, 1), drange_out=(0, 1)):
    """
    adjust the dynamic colour range of the given input data
    :param data: input image data
    :param drange_in: original range of input
    :param drange_out: required range of output
    :return: img => colour range adjusted images
    """
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
                np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return torch.clamp(data, min=0, max=1)

class G_imgs:
    def __init__(self,pretrained_G_path):
        self.opt=opt        

        self.args=CN()
        self.args.config='.configs/stylegan_512.yaml'
        self.args.pretrained_G_path=pretrained_G_path # pretrained weights file for generator
        self.args.num_samples=1 # number of synchronized grids to be generated
        self.args.output_dir=None # path to the output directory for the frames
        self.args.input=None # the dlatent code (W) for a certain sample
        self.args.output=None # the output for the certain samples
        self.opt.update(self.args)     
        # self.opt.freeze()
        
    # Load fewer layers of pre-trained models if possible
    def load(self,model, cpk_file):
        pretrained_dict = torch.load(cpk_file)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    
    def generate_imgs(self):
        print("Creating generator object ...")
        gen = Generator(resolution=opt.dataset.resolution,
                        num_channels=opt.dataset.channels,
                        structure=opt.structure,
                        **self.opt.model.gen)        
        print("Loading the generator weights from:", self.opt.pretrained_G_path)
        # load the weights into it
        #gen.load_state_dict(torch.load(self.opt.pretrained_G_path))      
        self.load(gen,self.opt.pretrained_G_path)
        
        # path for saving the files:
        save_path =self.opt.output_dir
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            
        latent_size = self.opt.model.gen.latent_size
        out_depth = int(np.log2(self.opt.dataset.resolution)) - 2          
        
        imgs_lst=[]
        if self.opt.input is None:
            print("Generating scale synchronized images ...")
            for img_num in tqdm(range(1, self.opt.num_samples + 1)):
                # generate the images:
                with torch.no_grad():
                    point = torch.randn(1, latent_size)
                    point = (point / point.norm()) * (latent_size ** 0.5)
                    ss_image = gen(point, depth=out_depth, alpha=1)
                    # color adjust the generated image:
                    ss_image = adjust_dynamic_range(ss_image)
    
                # save the ss_image in the directory
                if save_path:
                    save_image(ss_image, os.path.join(save_path, str(img_num) + ".png"))                      
                else: 
                    imgs_lst.append(ss_image)               

        else:
            code = np.load(self.opt.input)
            dlatent_in = torch.unsqueeze(torch.from_numpy(code), 0)
            ss_image = gen.g_synthesis(dlatent_in, depth=out_depth, alpha=1)
            # color adjust the generated image:
            ss_image = adjust_dynamic_range(ss_image)
            if save_path:
               save_image(ss_image, self.opt.output)    
            else:
                imgs_lst.append(ss_image)
                
        if save_path:
            print("Generated %d images at %s" % (self.opt.num_samples, save_path))
        else:
            return imgs_lst
                    
                      
if __name__ == '__main__':
    pass

