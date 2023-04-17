# -*- coding: utf-8 -*-
from yacs.config import CfgNode as CN

if __package__:    
    from ._base_options import cfg as opt
else:
    from _base_options import cfg as opt

cfg = CN()
cfg.update(opt)   

# train parameters
cfg.test = CN()
cfg.test.results_dir = './results/' # saves results here.
cfg.test.aspect_ratio = 1.0 # aspect ratio of result images
# Dropout and Batchnorm has different behavioir during training and test.
cfg.test.eval = False # use eval mode during test time.
cfg.test.num_test = 50 # how many test images to run
cfg.model.model ='test' # rewrite devalue values
# To avoid cropping, the load_size should be the same as crop_size
cfg.dataset.load_size = cfg.dataset.crop_size
cfg.basic.isTrain = False # train or test
cfg.test.model_suffix = '' # In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.

cfg.train = CN()
cfg.train.saveload = CN()
cfg.train.saveload.phase = 'test' # train, val, test, etc
cfg.train.visual = CN()
cfg.train.visual.display_id = -1 