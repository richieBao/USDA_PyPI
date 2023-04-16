import os
import argparse
import numpy as np
from PIL import Image

import torch

from .models._GAN import Generator
from ._generate_grid import adjust_dynamic_range
from ._config import cfg as opt
from yacs.config import CfgNode as CN
        
class G_depth_mixing_imgs:
    def __init__(self,pretrained_G_path):
        self.opt=opt        

        self.args=CN()
        self.args.config='.configs/stylegan_512.yaml'        
        self.args.pretrained_G_path=pretrained_G_path # pretrained weights file for generator
        self.args.output_dir=None # path to the output directory for the frames
        self.args.png_name='figure08-truncation-trick.png'
        self.args.out_depth=5
        self.args.src_seeds=[639, 1995, 687, 615, 1999]
        self.args.dst_seeds=[888, 888, 888]
        self.args.style_ranges=[range(0, 2)] * 1 + [range(2, 8)] * 1 + [range(8, 10)] * 1
        self.args.src_psis=[-0.5,-0.5,-0.5,-0.5,-0.5]
        self.args.dst_psis=[0.7,0.7,0.7]
        
        self.opt.update(self.args)        
        
        self.gen=self.load_gen()
        
    def load_gen(self):
        print("Creating generator object ...")
        gen = Generator(resolution=opt.dataset.resolution,
                        num_channels=opt.dataset.channels,
                        structure=opt.structure,
                        **self.opt.model.gen)        
        print("Loading the generator weights from:", self.opt.pretrained_G_path)
        # load the weights into it      
        self.load(gen,self.opt.pretrained_G_path)
        
        return gen
        
    # Load fewer layers of pre-trained models if possible
    def load(self,model, cpk_file):
        pretrained_dict = torch.load(cpk_file)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
    def draw_depth_mixing_figure(self):
        n_col = len(self.opt.src_seeds)
        n_row = len(self.opt.dst_seeds)
        w = h = 2 ** (self.opt.out_depth + 2)
        with torch.no_grad():
            latent_size = self.gen.g_mapping.latent_size
            src_latents_np = np.stack([np.random.RandomState(seed).randn(latent_size, ) for seed in self.opt.src_seeds])
            dst_latents_np = np.stack([np.random.RandomState(seed).randn(latent_size, ) for seed in self.opt.dst_seeds])           
            
            src_latents = torch.from_numpy(src_latents_np.astype(np.float32))
            dst_latents = torch.from_numpy(dst_latents_np.astype(np.float32))
            
            
            src_dlatents = self.gen.g_mapping(src_latents)  # [seed, layer, component]
            dst_dlatents = self.gen.g_mapping(dst_latents)  # [seed, layer, component]
            
            # print(src_dlatents.shape,dst_dlatents.shape,)
            #------------------------------------------------------------------
            dlatent_avg = self.gen.truncation.avg_latent.numpy() 

            src_dlatents = src_dlatents.detach().numpy() 
            dst_dlatents = dst_dlatents.detach().numpy() 
            
            src_dlatents = (src_dlatents - dlatent_avg) * np.reshape(self.opt.src_psis, [-1, 1,1]) + dlatent_avg
            dst_dlatents = (dst_dlatents - dlatent_avg) * np.reshape(self.opt.dst_psis, [-1, 1,1]) + dlatent_avg
            
            src_dlatents = torch.from_numpy(src_dlatents.astype(np.float32))
            dst_dlatents = torch.from_numpy(dst_dlatents.astype(np.float32))
            #------------------------------------------------------------------
            # print(src_dlatents.shape,dst_dlatents.shape,dlatent_avg.shape)            
            
            
            src_images = self.gen.g_synthesis(src_dlatents, depth=self.opt.out_depth, alpha=1)
            dst_images = self.gen.g_synthesis(dst_dlatents, depth=self.opt.out_depth, alpha=1)
    
            src_dlatents_np = src_dlatents.numpy()
            dst_dlatents_np = dst_dlatents.numpy()
            canvas = Image.new('RGB', (w * (n_col + 1), h * (n_row + 1)), 'white')
            for col, src_image in enumerate(list(src_images)):
                src_image = adjust_dynamic_range(src_image)
                src_image = src_image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
                canvas.paste(Image.fromarray(src_image, 'RGB'), ((col + 1) * w, 0))
            for row, dst_image in enumerate(list(dst_images)):
                dst_image = adjust_dynamic_range(dst_image)
                dst_image = dst_image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
                canvas.paste(Image.fromarray(dst_image, 'RGB'), (0, (row + 1) * h))
    
                row_dlatents = np.stack([dst_dlatents_np[row]] * n_col)
                row_dlatents[:, self.opt.style_ranges[row]] = src_dlatents_np[:, self.opt.style_ranges[row]]
                row_dlatents = torch.from_numpy(row_dlatents)
    
                row_images = self.gen.g_synthesis(row_dlatents, depth=self.opt.out_depth, alpha=1)
                for col, image in enumerate(list(row_images)):
                    image = adjust_dynamic_range(image)
                    image = image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
                    canvas.paste(Image.fromarray(image, 'RGB'), ((col + 1) * w, (row + 1) * h))
                    
            if self.opt.output_dir:
                canvas.save(self.opt.png_name)        
            else:
                return canvas
        print('Done.')    

if __name__ == '__main__':
    pass
