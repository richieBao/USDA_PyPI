# -*- coding: utf-8 -*-
if __package__:
    from yacs.config import CfgNode as CN
else:
    from yacs.config import CfgNode as CN

cfg = CN()

# basic parameters
cfg.basic = CN()
cfg.basic.dataroot='' # path to images (should have subfolders trainA, trainB, valA, valB, etc)
cfg.basic.name='' # name of the experiment. It decides where to store samples and models
cfg.basic.gpu_ids=[0] # gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU
cfg.basic.checkpoints_dir='./checkpoints' # models are saved here
cfg.basic.isTrain = True # train or test

# model parameters
cfg.model = CN()
cfg.model.model ='pix2pix'  # chooses which model to use. [cycle_gan | pix2pix | test | colorization]
cfg.model.input_nc = 3 # of input image channels: 3 for RGB and 1 for grayscale
cfg.model.output_nc = 3 # of output image channels: 3 for RGB and 1 for grayscale
cfg.model.ngf = 64 # of gen filters in the last conv layer
cfg.model.ndf = 64 # of discrim filters in the first conv layer
cfg.model.netD = 'basic' # specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator'
cfg.model.netG = 'unet_256' # specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]
cfg.model.n_layers_D = 3 # only used if netD==n_layers
cfg.model.norm = 'batch' # instance normalization or batch normalization [instance | batch | none]
cfg.model.init_type = 'normal' # network initialization [normal | xavier | kaiming | orthogonal]
cfg.model.init_gain = 0.02 # scaling factor for normal, xavier and orthogonal.
cfg.model.no_dropout = True # no dropout for the generator

# dataset parameters
cfg.dataset = CN()
cfg.dataset.dataset_mode = 'aligned'  # chooses how datasets are loaded. [unaligned | aligned | single | colorization]
cfg.dataset.direction = 'AtoB'  # AtoB or BtoA
cfg.dataset.serial_batches = False  # if true, takes images in order to make batches, otherwise takes them randomly
cfg.dataset.num_threads = 4  # threads for loading data
cfg.dataset.batch_size = 1  # input batch size
cfg.dataset.load_size = 286 # scale images to this size
cfg.dataset.crop_size = 256  # then crop to this size
cfg.dataset.max_dataset_size = float("inf")  # Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.
cfg.dataset.preprocess = 'resize_and_crop'  # 'scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]
cfg.dataset.no_flip = True  # if specified, do not flip the images for data augmentation
cfg.dataset.display_winsize = 256  # display window size for both visdom and HTML

# additional parameters
cfg.additional = CN()
cfg.additional.epoch = 'latest' # which epoch to load? set to latest to use latest cached model
cfg.additional.load_iter = 0 # which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]
cfg.additional.verbose = False # if specified, print more debugging information
cfg.additional.suffix = '' # customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}

# wandb parameters
cfg.wandb = CN()
cfg.wandb.use_wandb = False  # if specified, then init wandb logging
cfg.wandb.wandb_project_name = 'pix2pix' # specify wandb project name
