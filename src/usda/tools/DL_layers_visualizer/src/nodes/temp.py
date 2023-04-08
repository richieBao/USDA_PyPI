from pathlib import Path
from fastai.vision.all import *
from fastai.vision.gan import *
import matplotlib.pyplot as plt
from fastai.vision.gan import *

def cal_imgs_batch(imgs_folder):      

    imgs_folder=Path(imgs_folder)
    bs=128
    size=64  
    files=get_image_files(imgs_folder)
    
    def label_func(f): return 1
    # ngpu=1
    # device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    dls=ImageDataLoaders.from_name_func(imgs_folder, files, label_func,item_tfms=Resize(size),bs=bs)  # ,device=device
    x,_=dls.one_batch()
    return x


def gan_model(model_path,imgs_folder,x):
    size=64
    bs=128
    
    dblock=DataBlock(
        blocks=(TransformBlock, ImageBlock),
        get_x=generate_noise,
        get_items=get_image_files,
        splitter=IndexSplitter([]),
        item_tfms=Resize(size, method=ResizeMethod.Crop), 
        batch_tfms=Normalize.from_stats(torch.tensor([0.5,0.5,0.5]), torch.tensor([0.5,0.5,0.5])))    
    
    dls=dblock.dataloaders(imgs_folder, path=imgs_folder, bs=bs)
    generator=basic_generator(64, n_channels=3, n_extra_layers=1)
    critic=basic_critic(64, n_channels=3, n_extra_layers=1, act_cls=partial(nn.LeakyReLU, negative_slope=0.2))    
    
    generator=basic_generator(64, n_channels=3, n_extra_layers=1)
    critic=basic_critic(64, n_channels=3, n_extra_layers=1, act_cls=partial(nn.LeakyReLU, negative_slope=0.2))
    
    learn=GANLearner.wgan(dls, generator, critic, opt_func=RMSProp,
                          lr=1e-5, 
                          clip=0.01,
                          switcher=FixedGANSwitcher(n_crit=5, n_gen=1),
                          switch_eval=False
                          )    
    
    learn.recorder.train_metrics=True
    learn.recorder.valid_metrics=False    
    learn.load(model_path,with_opt=True)    
   
    netD=learn.model.critic
    netG=learn.model.generator

    # batch_tfms=Normalize.from_stats(torch.tensor([0.5,0.5,0.5]), torch.tensor([0.5,0.5,0.5]))
    ngpu=1
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    netD=netD.to(torch.device('cuda'))
    netG=netG.to(torch.device('cuda'))
    
    results=netD(x).detach().cpu()

    counts, bin_edges = np.histogram(results, bins=10,density=True)
    pdf=counts/sum(counts)
    
    nz=100 
    fixed_noise = torch.randn(x.shape[0], nz,device=device)
    fake=netG(fixed_noise)
    fake_results=netD(fake).detach().cpu()
    fake_counts, fake_bin_edges = np.histogram(fake_results, bins=10,density=True)
    fake_pdf=fake_counts/sum(fake_counts)
    
    
    plt.plot(bin_edges[1:],pdf,label="real_normalised_pdf")
    plt.plot(bin_edges[1:],counts,label="real_actual_pdf")
    
    plt.plot(fake_bin_edges[1:],fake_pdf,label="fake_normalised_pdf")
    plt.plot(fake_bin_edges[1:],fake_counts,label="fake_actual_pdf")
    plt.show()
    
    
    
    
    
 
if __name__=="__main__":
    imgs_folder=r'I:\data\NAIP4StyleGAN\patches_64'
    model_path=r'C:\Users\richi\omen_richiebao\omen_github\USDA_PyPI\src\usda\tools\DL_layers_visualizer\naip_wgan_c_learn'
    a=cal_imgs_batch(imgs_folder)      
    gan_model(model_path,imgs_folder,a)