# -*- coding: utf-8 -*-
from yacs.config import CfgNode as CN

if __package__:    
    from ._base_options import cfg as opt
else:
    from _base_options import cfg as opt

cfg = CN()
cfg.update(opt) 

# train parameters
cfg.train = CN()
cfg.train.n_epochs = 100 # number of epochs with the initial learning rate
cfg.train.n_epochs_decay = 100 # number of epochs to linearly decay learning rate to zero
cfg.train.beta1 = 0.5 # momentum term of adam
cfg.train.lr = 0.0002 # initial learning rate for adam
cfg.train.gan_mode = 'vanilla' # the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.
cfg.train.pool_size = 0 # the size of image buffer that stores previously generated images (50)
cfg.train.lr_policy = 'linear' # learning rate policy. [linear | step | plateau | cosine]
cfg.train.lr_decay_iters = 50 # multiply by a gamma every lr_decay_iters iterations
cfg.train.lambda_L1 = 100.0 # weight for L1 loss

# network saving and loading parameters
cfg.train.saveload = CN()
cfg.train.saveload.save_latest_freq = 5000 # frequency of saving the latest results
cfg.train.saveload.save_epoch_freq = 5 # frequency of saving checkpoints at the end of epochs
cfg.train.saveload.save_by_iter = False # whether saves model by iteration
cfg.train.saveload.continue_train = False # continue training: load the latest model
cfg.train.saveload.epoch_count = 1 # the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...
cfg.train.saveload.phase = 'train' # train, val, test, etc

# visdom and HTML visualization parameters
cfg.train.visual = CN()
cfg.train.visual.display_freq = 400 # frequency of showing training results on screen
cfg.train.visual.display_ncols = 4 # if positive, display all images in a single visdom web panel with certain number of images per row.
cfg.train.visual.display_id = 1 # window id of the web display
cfg.train.visual.display_server = "http://localhost" # visdom server of the web display
cfg.train.visual.display_env = 'main' # visdom display environment name (default is "main")
cfg.train.visual.display_port = 8097 # visdom port of the web display
cfg.train.visual.update_html_freq = 1000 # frequency of saving training results to html
cfg.train.visual.print_freq = 100 # frequency of showing training results on console
cfg.train.visual.no_html = False # do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/

  