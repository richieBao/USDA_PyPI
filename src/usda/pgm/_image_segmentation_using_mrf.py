# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 11:48:43 2023

@author: richie bao
ref:Markov Random Fields, W.G.H.S. Weligampola (E/14/379),June 2020
"""
from PIL import Image
import numpy as np
from scipy.cluster import vq
from scipy import signal
import cv2 as cv
import scipy
import matplotlib.pyplot as plt

class Image_segmentation_using_MRF:
    # Image segmentation using MRF model
    
    def __init__(self,img_fn,nlevels=4,beta=0.5,std=7,win_dim=255,limit_dim=7,imshow=True,figsize=(20,10)):
        self.beta=0.5
        self.std=7
        self.figsize=figsize
        
        # Read in image
        im=Image.open(img_fn)
        im=np.array(im)
        # If grayscale add one dimension for easy processing
        if im.ndim==2:
            im=im[:,:,newaxis]
            
        # Initial kmean segmentation
        lev=self.getinitkmean(im,nlevels)
        self.original_lev=np.copy(lev)
        
        # MRF ICM
        win_dim=win_dim
        while (win_dim>limit_dim): # 7
            print(win_dim)
            locav=self.local_average(im,lev,nlevels,win_dim)
            # print(locav.shape)
            lev=self.MRF(im,lev,locav,nlevels) 
            win_dim=win_dim//2
            
        self.lev=lev
        self.locav=locav
        
        if imshow:
            self.im_show(lev,locav)      
        
    def getinitkmean(self,im,nlevels):
        obs=im.reshape(im.shape[0]*im.shape[1],-1)
        obs=vq.whiten(obs) # Normalize a group of observations on a per feature basis.
        (centroids,lev)=vq.kmeans2(obs, nlevels)
        lev=lev.reshape(im.shape[0],im.shape[1])
        
        return lev  
    
    def local_average(self,obs,lev,nlevels,win_dim):
        # Use correlation to perform averaging
        mask=np.ones((win_dim,win_dim))/win_dim**2
        
        # 4d array (512,512,ncolors,nlevels)
        locav=np.ones((obs.shape+(nlevels,))) # (512, 512, 3, 4)
        # print(obs.shape,nlevels)
        for i in range(obs.shape[2]):
            for j in range(nlevels):
                temp=(obs[:,:,i]*(lev==j))
                locav[:,:,i,j]=signal.fftconvolve(temp,mask,mode='same')
         
        return locav
    
    def MRF(self,obs,lev,locav,nlevels):
        (M,N)=obs.shape[0:2]
        for i in range(M):
            for j in range(N):
                # Find segmentation level which has min energy (highest posterior)
                cost=[self.energy(k,i,j,obs,lev,locav) for k in range(nlevels)]
                lev[i,j]=cost.index(min(cost))   
                
        return lev     
                
    def energy(self,pix_lev,i,j,obs,lev,locav):
        cl=self.clique(pix_lev,i,j,lev)
        closeness=np.linalg.norm(locav[i,j,:,pix_lev]-obs[i,j,:])
        # print(locav[i,j,:,pix_lev].shape,obs[i,j,:].shape)
        return self.beta*cl+closeness/self.std**2
        
    def clique(self,pix_lev,i,j,lev):
        (M,N)=lev.shape[0:2]
        
        # find correct neighbors
        if (i==0 and j==0):
            neighbor=[(0,1),(1,0)]
        elif i==0 and j==N-1:
            neighbor=[(0,N-2),(1,N-1)]
        elif i==M-1 and j==0:
            neighbor=[(M-1,1),(M-2,0)]
        elif i==M-1 and j==N-1:
            neighbor=[(M-1,N-2),(M-2,N-1)]
        elif i==0:
            neighbor=[(0,j-1),(0,j+1),(1,j)]
        elif i==M-1:
            neighbor=[(M-1,j-1),(M-1,j+1),(M-2,j)]
        elif j==0:
            neighbor=[(i-1,0),(i+1,0),(i,1)]
        elif j==N-1:
            neighbor=[(i-1,N-1),(i+1,N-1),(i,N-2)]
        else:
            neighbor=[(i-1,j),(i+1,j),(i,j-1),(i,j+1),(i-1,j-1),(i-1,j+1),(i+1,j-1),(i+1,j+1)]
            
        return sum(self.delta(pix_lev,lev[i]) for i in neighbor)
        
    def delta(self,a,b):
        if a==b:
            return -1
        else:
            return 1
    
    def im_show(self,lev,locav):        
        f, axes=plt.subplots(1,2,figsize=self.figsize)
        axes[0].imshow(lev*20)        
        
        out=self.ACAreconstruction(lev,locav)
        axes[1].imshow(out/np.max(out))
        
        axes[0].set_title('Level')
        axes[1].set_title('Seg Image')
        plt.show()
        
    def ACAreconstruction(self,lev,locav):
        out=0
        for i in range(locav.shape[3]):
            mask=(lev==i)
            out+=mask[:,:,np.newaxis]*locav[:,:,:,i]
            
        return out


if __name__=="__main__":
    img_fn=r'C:\Users\richi\omen_richiebao\omen_github\USDA_special_study\imgs\3_5_d\3_5_d_01.jpg'
    seg=Image_segmentation_using_MRF(img_fn,nlevels=7,win_dim=64)