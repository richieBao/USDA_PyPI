"""
Author: richie bao

migrated from:
-------------------------------------------------
   File Name:    train.py
   Author:       Zhonghao Huang
   Date:         2019/10/18
   Description:
-------------------------------------------------
"""
import argparse
import os
import shutil
from yacs.config import CfgNode as CN
from pathlib import Path
import yaml

import torch
from torch.backends import cudnn
from ._config import cfg as opt

from .data import make_dataset
from .models import StyleGAN
from .utils import (copy_files_and_create_dirs,list_dir_recursively_with_ignore, make_logger)

from yacs.config import CfgNode as CN
args=CN()
args.ckpt=CN()
args.ckpt.start_depth= 0
args.ckpt.generator_file=None
args.ckpt.gen_shadow_file=None
args.ckpt.discriminator_file=None
args.ckpt.gen_optim_file=None
args.ckpt.dis_optim_file=None

# Load fewer layers of pre-trained models if possible
def load(model, cpk_file):
    pretrained_dict = torch.load(cpk_file)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)   

class Stylegan_train:
    
    def __init__(self):
        pass
        self.args=args
        self.opt=opt
        self.opt.update(self.args)           
        
    def opt_freeze_or_defrost(self,opt_freeze=True):
        if opt_freeze==True:
            self.opt.freeze()
        else:
            self.opt.defrost()
    
    def configuration(self):
        # make output dir
        output_dir = self.opt.output_dir
        if os.path.exists(output_dir):
            raise KeyError("Existing path: ", output_dir)
        os.makedirs(output_dir)
        
        # copy codes and config file
        package_path=Path(os.path.realpath(__file__)).parent
        files = list_dir_recursively_with_ignore(package_path, ignores=['diagrams', 'configs'])
        files = [(f[0], os.path.join(output_dir, "src", f[1])) for f in files]
        copy_files_and_create_dirs(files)          
        with open(Path(output_dir)/'opt.yaml', 'w') as outfile:
            yaml.dump(self.opt.dump(), outfile, default_flow_style=False)
    
        # logger
        self.logger = make_logger("project", self.opt.output_dir, 'log')

        # device
        if self.opt.device == 'cuda':
            os.environ['CUDA_VISIBLE_DEVICES'] = self.opt.device_id
            num_gpus = len(self.opt.device_id.split(','))
            self.logger.info("Using {} GPUs.".format(num_gpus))
            self.logger.info("Training on {}.\n".format(torch.cuda.get_device_name(0)))
            cudnn.benchmark = True
        self.device = torch.device(self.opt.device)  
        
    def build_dataset(self):
        # create the dataset for training
        self.dataset = make_dataset(self.opt.dataset, conditional=self.opt.conditional)   
        
        return self.dataset

    def init_network(self):
        self.style_gan = StyleGAN(structure=self.opt.structure,
                             conditional=self.opt.conditional,
                             n_classes=self.opt.n_classes,
                             resolution=self.opt.dataset.resolution,
                             num_channels=self.opt.dataset.channels,
                             latent_size=self.opt.model.gen.latent_size,
                             g_args=self.opt.model.gen,
                             d_args=self.opt.model.dis,
                             g_opt_args=self.opt.model.g_optim,
                             d_opt_args=self.opt.model.d_optim,
                             loss=self.opt.loss,
                             drift=self.opt.drift,
                             d_repeats=self.opt.d_repeats,
                             use_ema=self.opt.use_ema,
                             ema_decay=self.opt.ema_decay,
                             device=self.device)
    
        # Resume training from checkpoints
        if self.args.ckpt.generator_file is not None:
            self.logger.info("Loading generator from: %s", self.args.ckpt.generator_file)
            load(style_gan.gen, self.args.ckpt.generator_file)
        else:
            self.logger.info("Training from scratch...")
    
        if self.args.ckpt.discriminator_file is not None:
            self.logger.info("Loading discriminator from: %s", self.args.ckpt.discriminator_file)
            style_gan.dis.load_state_dict(torch.load(self.args.ckpt.discriminator_file))
    
        if self.args.ckpt.gen_shadow_file is not None and self.opt.use_ema:
            self.logger.info("Loading shadow generator from: %s", self.args.ckpt.gen_shadow_file)
            load(style_gan.gen_shadow, self.args.ckpt.gen_shadow_file)
    
        if self.args.ckpt.gen_optim_file is not None:
            self.logger.info("Loading generator optimizer from: %s", self.args.ckpt.gen_optim_file)
            style_gan.gen_optim.load_state_dict(torch.load(self.args.ckpt.gen_optim_file))
    
        if self.args.ckpt.dis_optim_file is not None:
            self.logger.info("Loading discriminator optimizer from: %s", self.args.ckpt.dis_optim_file)
            style_gan.dis_optim.load_state_dict(torch.load(self.args.ckpt.dis_optim_file))    
            
            
    def train(self):
        # train the network
        self.style_gan.train(dataset=self.dataset,
                      num_workers=self.opt.num_works,
                      epochs=self.opt.sched.epochs,
                      batch_sizes=self.opt.sched.batch_sizes,
                      fade_in_percentage=self.opt.sched.fade_in_percentage,
                      logger=self.logger,
                      output=self.opt.output_dir,
                      num_samples=self.opt.num_samples,
                      start_depth=self.args.ckpt.start_depth,
                      feedback_factor=self.opt.feedback_factor,
                      checkpoint_factor=self.opt.checkpoint_factor)        
                    
                        
            

