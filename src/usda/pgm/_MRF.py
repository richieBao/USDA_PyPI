# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 12:42:40 2023

@author: richie bao
"""
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from torch.optim import RMSprop

def MRF_binary(I,J,eta=2.0,zeta=1.5):
    I=np.copy(I)
    J=np.copy(J)
    
    ind=np.arange(np.shape(I)[0])
    np.random.shuffle(ind)
    orderx = ind.copy()
    np.random.shuffle(ind)

    for i in tqdm(orderx):
        for j in ind:
            oldJ = J[i,j]
            J[i,j]=1
            patch = 0
            for k in range(-1,1):
                for l in range(-1,1):
                    patch += J[i,j] * J[i+k,j+l]
                 
            energya = -eta*np.sum(I*J) - zeta*patch

            J[i,j]=-1
            patch = 0
            for k in range(-1,1):
                for l in range(-1,1):
                    patch += J[i,j] * J[i+k,j+l]
                    
            energyb = -eta*np.sum(I*J) - zeta*patch
            
            if energya<energyb:
                J[i,j] = 1
            else:
                J[i,j] = -1
                
    return J

class MRF_continuous:
    
    def __init__(self,noisy_img_fn,alpha=0.7,channel=0,epochs=100,original_img_fn=None,cuda=False):        
        self.epochs=epochs
        self.alpha=alpha
        
        to_tensor=ToTensor()
        self.noisy=to_tensor(Image.open(noisy_img_fn))[channel] 
        self.X=torch.zeros_like(self.noisy)
        
        if original_img_fn:
            self.original_img=to_tensor(Image.open(original_img_fn))[channel]                     
                
        if cuda:
            self.noisy=self.noisy.cuda()
            self.X=self.X.cuda()
            if original_img_fn:
                self.original_img=self.original_img.cuda()
        
        self.RRMSE = lambda x,y: (((y - x)**2).sum() / (y**2).sum())**0.5
        self.X.requires_grad = True
        self.optimizer=RMSprop([self.X]) 
        
    def mrf_prior(self,x):
        
        return x**2
    
    def mrf_loss(self,X, noisy, alpha):
        loss1 = ((noisy - X)**2).sum()
        loss2 = 0
        loss2 += self.mrf_prior(X[:, 1: ] - X[:, :-1]).sum()
        loss2 += self.mrf_prior(X[:-1, :] - X[ 1:, :]).sum()
        
        return alpha*loss1 + 2*loss2    
    
    def train(self):
        self.errors = []
        self.losses = []
        self.images = []         
        
        for it in tqdm(range(self.epochs)):            
            self.optimizer.zero_grad()
            loss = self.mrf_loss(self.X, self.noisy, self.alpha)
            loss.backward()
            self.optimizer.step()
            
            if self.original_img is not None:
                self.errors.append(self.RRMSE(self.X,self.original_img))
                
            self.losses.append(loss.item())
            self.images.append(np.array(255*self.X.clone().detach().cpu()).astype(np.uint8))  