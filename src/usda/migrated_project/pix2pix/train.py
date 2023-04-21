# -*- coding: utf-8 -*-
"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
from yacs.config import CfgNode as CN
import time

if __package__:    
    from .options import cfg_train 
    from .data import create_dataset
    from .models import create_model
    from .util._visualizer import Visualizer
else:
    from options import cfg_train 
    from data import create_dataset
    from models import create_model
    from util._visualizer import Visualizer

class Pix2pix_train:
    def __init__(self):    
        self.opt=cfg_train # get training options
        self.total_iters = 0                # the total number of training iterations
        
    def create_dataset(self):               
        self.dataset=create_dataset(self.opt) # create a dataset given opt.dataset_mode and other options
        self.dataset_size = len(self.dataset) # get the number of images in the dataset.
        print('The number of training images = %d' % self.dataset_size)
        
    def create_model(self):
        self.model = create_model(self.opt)      # create a model given opt.model and other options
        self.model.setup(self.opt)               # regular setup: load and print networks; create schedulers
        
    def visualizer(self):
        self.visualizer=Visualizer(self.opt)   # create a visualizer that display/save images and plots
        
    def train(self):
        for epoch in range(self.opt.train.saveload.epoch_count, self.opt.train.n_epochs + self.opt.train.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()    # timer for data loading per iteration
            epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
            self.visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
            self.model.update_learning_rate()    # update learning rates in the beginning of every epoch.            
            for i, data in enumerate(self.dataset):  # inner loop within one epoch
                iter_start_time = time.time()  # timer for computation per iteration                
                if self.total_iters % self.opt.train.visual.print_freq == 0:
                    t_data = iter_start_time - iter_data_time           
                    
                self.total_iters += self.opt.dataset.batch_size
                epoch_iter += self.opt.dataset.batch_size            

                self.model.set_input(data)         # unpack data from dataset and apply preprocessing
                self.model.optimize_parameters()   # calculate loss functions, get gradients, update network weights 
                
                if self.total_iters % self.opt.train.visual.display_freq == 0:   # display images on visdom and save images to a HTML file
                    save_result = self.total_iters % self.opt.train.visual.update_html_freq == 0
                    self.model.compute_visuals()
                    self.visualizer.display_current_results(self.model.get_current_visuals(), epoch, save_result)     
                    
                if self.total_iters % self.opt.train.visual.print_freq == 0:    # print training losses and save logging information to the disk
                    losses = self.model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / self.opt.dataset.batch_size
                    self.visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                    if self.opt.train.visual.display_id > 0:
                        self.visualizer.plot_current_losses(epoch, float(epoch_iter) / self.dataset_size, losses)                    
                        
                if self.total_iters % self.opt.train.saveload.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                    print('saving the latest model (epoch %d, total_iters %d)' % (epoch, self.total_iters))
                    save_suffix = 'iter_%d' % self.total_iters if self.opt.train.saveload.save_by_iter else 'latest'
                    self.model.save_networks(save_suffix)                        
                            
                iter_data_time = time.time()            

            if epoch % self.opt.train.saveload.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
                print('saving the model at the end of epoch %d, iters %d' % (epoch, self.total_iters))
                self.model.save_networks('latest')
                self.model.save_networks(epoch)
                
            print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, self.opt.train.n_epochs + self.opt.train.n_epochs_decay, time.time() - epoch_start_time))
        
if __name__=="__main__":
    p2p=Pix2pix_train()
    
    p2p.opt.basic.dataroot=r'I:\data\pix2pix_dataset\maps'
    p2p.opt.dataset.dataset_mode='aligned'    
    p2p.opt.dataset.direction='BtoA'
    p2p.create_dataset()
    dl=p2p.dataset
    
    p2p.opt.basic.checkpoints_dir=r'I:\model_ckpts\pix2pix_02'
    p2p.create_model()
    m=p2p.model
    
    # p2p.opt.train.visual.display_id=-1
    # p2p.opt.wandb.use_wandb=True
    p2p.visualizer()
    
    p2p.train()
    
    




