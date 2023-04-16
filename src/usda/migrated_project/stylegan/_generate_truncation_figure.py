"""
-------------------------------------------------
   File Name:    generate_truncation_figure.py
   Author:       Zhonghao Huang
   Date:         2019/11/23
   Description:  
-------------------------------------------------
"""
import numpy as np
from PIL import Image
import torch

from ._generate_grid import adjust_dynamic_range
from .models._GAN import Generator
from ._config import cfg as opt
from yacs.config import CfgNode as CN

class G_truncation_imgs:
    def __init__(self,pretrained_G_path):
        self.opt=opt        

        self.args=CN()
        self.args.config='.configs/stylegan_512.yaml'        
        self.args.pretrained_G_path=pretrained_G_path # pretrained weights file for generator
        self.args.output_dir=None # path to the output directory for the frames
        self.args.png_name='figure08-truncation-trick.png'
        self.args.out_depth=5
        self.args.seeds=[91, 388]
        self.args.psis=[1, 0.7, 0.5, 0, -0.5, -1]
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
        
    def draw_truncation_trick_figure(self):
        w = h = 2 ** (self.opt.out_depth + 2)
        latent_size = self.gen.g_mapping.latent_size
    
        with torch.no_grad():
            latents_np = np.stack([np.random.RandomState(seed).randn(latent_size) for seed in self.opt.seeds])
            latents = torch.from_numpy(latents_np.astype(np.float32))
            dlatents = self.gen.g_mapping(latents).detach().numpy()  # [seed, layer, component]
            dlatent_avg = self.gen.truncation.avg_latent.numpy()  # [component]
    
            canvas = Image.new('RGB', (w * len(self.opt.psis), h * len(self.opt.seeds)), 'white')
            for row, dlatent in enumerate(list(dlatents)):
                row_dlatents = (dlatent[np.newaxis] - dlatent_avg) * np.reshape(self.opt.psis, [-1, 1, 1]) + dlatent_avg
                row_dlatents = torch.from_numpy(row_dlatents.astype(np.float32))
                row_images = self.gen.g_synthesis(row_dlatents, depth=self.opt.out_depth, alpha=1)
                for col, image in enumerate(list(row_images)):
                    image = adjust_dynamic_range(image)
                    image = image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
                    canvas.paste(Image.fromarray(image, 'RGB'), (col * w, row * h))
            
            if self.opt.output_dir:
                canvas.save(self.opt.png_name)        
            else:
                return canvas
        print('Done.')   


if __name__ == '__main__':
    pass
