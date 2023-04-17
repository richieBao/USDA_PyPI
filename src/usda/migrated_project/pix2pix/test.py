"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os

if __package__:
    from .options import cfg_test
    from .data import create_dataset
    from .models import create_model
    from .util import save_images
    from .util import HTML
else:
    from options import cfg_test
    from data import create_dataset
    from models import create_model
    from util import save_images
    from util import HTML

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

class Pix2pix_test:
    def __init__(self):    
        self.opt=cfg_test # get training options
        # hard-code some parameters for test
        self.opt.dataset.num_threads = 0   # test code only supports num_threads = 0
        self.opt.dataset.batch_size = 1    # test code only supports batch_size = 1
        self.opt.dataset.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        self.opt.dataset.no_flip = True    # no flip; comment this line if results on flipped images are needed.
        self.opt.train.visual.display_id = -1   # no visdom display; the test code saves the results to a HTML file.        
    
    def create_dataset(self):               
        self.dataset=create_dataset(self.opt) # create a dataset given opt.dataset_mode and other options

    def create_model(self):
        self.model = create_model(self.opt)      # create a model given opt.model and other options
        self.model.setup(self.opt)               # regular setup: load and print networks; create schedulers    

    def test(self):
        # initialize logger
        if self.opt.wandb.use_wandb:
            wandb_run = wandb.init(project=self.opt.wandb.wandb_project_name, name=self.opt.wandb.name, config=opt) if not wandb.run else wandb.run
            wandb_run._label(repo='CycleGAN-and-pix2pix')        
            
        # create a website
        web_dir = os.path.join(self.opt.test.results_dir, self.opt.basic.name, '{}_{}'.format(self.opt.train.saveload.phase, self.opt.additional.epoch))  # define the website directory
        if self.opt.additional.load_iter > 0:  # load_iter is 0 by default
            web_dir = '{:s}_iter{:d}'.format(web_dir, self.opt.additional.load_iter)
        print('creating web directory', web_dir)
        webpage = HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (self.opt.basic.name, self.opt.train.saveload.phase, self.opt.additional.epoch))
        # test with eval mode. This only affects layers like batchnorm and dropout.
        # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
        # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.        
        if self.opt.test.eval:
            self.model.eval()        
            
        for i, data in enumerate(self.dataset):
            if i >= self.opt.test.num_test:  # only apply our model to opt.num_test images.
                break            
            self.model.set_input(data)  # unpack data from data loader
            self.model.test()           # run inference
            visuals =self.model.get_current_visuals()  # get image results
            img_path = self.model.get_image_paths()     # get image paths

            if i % 5 == 0:  # save images to an HTML file
                print('processing (%04d)-th image... %s' % (i, img_path))

            save_images(webpage, visuals, img_path, aspect_ratio=self.opt.test.aspect_ratio, width=self.opt.dataset.display_winsize, use_wandb=self.opt.wandb.use_wandb)
         
        webpage.save()  # save the HTML
        

if __name__ == '__main__':
    p2p=Pix2pix_test()
    
    p2p.opt.basic.dataroot=r'I:\data\pix2pix_dataset\maps'
    p2p.opt.test.results_dir=r'I:\model_ckpts\pix2pix_02'
    p2p.opt.dataset.dataset_mode='single'#'aligned'    
    # p2p.opt.dataset.direction='AtoB'
    p2p.opt.train.saveload.phase='val'
    
    p2p.create_dataset()
    
    p2p.opt.basic.checkpoints_dir=r'I:\model_ckpts\pix2pix_02'
    p2p.create_model()
    m=p2p.model    

    p2p.test()
    
    
    