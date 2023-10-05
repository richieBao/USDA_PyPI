# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 17:15:19 2023

@author: richie bao

updated and transfered: Ising model, https://rajeshrinet.github.io/blog/2014/ising-model/
"""
# Simulating the Ising model
# from __future__ import division
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt

class Ising_Metropolis_MCMC():
    ''' Simulating the Ising model. Markov chain Monte Carlo (MCMC)  '''    
    ## monte carlo moves
    def __init__(self,N=64,temp=.4):
        self.N=N 
        self.temp=temp
    
    def mcmove(self, config, beta):
        ''' This is to execute the monte carlo moves using 
        Metropolis algorithm such that detailed
        balance condition is satisified'''
        N=self.N
        for i in range(N):
            for j in range(N):            
                    a = np.random.randint(0, N)
                    b = np.random.randint(0, N)
                    s =  config[a, b]                    
                    # print(a,b,[(a+1)%N,b],[a,(b+1)%N],[(a-1)%N,b],[a,(b-1)%N])
                    nb = config[(a+1)%N,b] + config[a,(b+1)%N] + config[(a-1)%N,b] + config[a,(b-1)%N]
                    cost = 2*s*nb
                    if cost < 0:	
                        s *= -1
                    elif rand() < np.exp(-cost*beta):
                        s *= -1
                    config[a, b] = s
        return config
    
    def simulate(self,msrmnt=1001,idxes=[1,4,32,100,1000],figsize=(15, 15)):   
        ''' This module simulates the Ising model'''
        # Initialse the lattice
        config = 2*np.random.randint(2, size=(self.N,self.N))-1        
        f = plt.figure(figsize=figsize, dpi=80);    
        self.configPlot(f, config, 0, 1);
                
        # msrmnt = 4#1001
        j=2
        for i in range(msrmnt):
            self.mcmove(config, 1.0/self.temp)
            if i in idxes:
                self.configPlot(f, config, i, j);
                j+=1
            # if i == 1:       self.configPlot(f, config, i, 2);
            # if i == 4:       self.configPlot(f, config, i, 3);
            # if i == 32:      self.configPlot(f, config, i, 4);
            # if i == 100:     self.configPlot(f, config, i, 5);
            # if i == 1000:    self.configPlot(f, config, i, 6);                 
                    
    def configPlot(self, f, config, i, n_):
        ''' This modules plts the configuration once passed to it along with time etc '''
        X, Y = np.meshgrid(range(self.N), range(self.N))
        sp =  f.add_subplot(3, 3, n_ )  
        plt.setp(sp.get_yticklabels(), visible=False)
        plt.setp(sp.get_xticklabels(), visible=False)      
        plt.pcolormesh(X, Y, config, cmap=plt.cm.RdBu.reversed());
        for y in range(config.shape[0]):
            for x in range(config.shape[1]):
                if config[y, x]==1:
                    plt.text(x, y, '+' ,
                             horizontalalignment='center',
                             verticalalignment='center',
                     )
                else:
                    plt.text(x, y, '-' ,
                             horizontalalignment='center',
                             verticalalignment='center',
                     )                   
        
        plt.title('Time=%d'%i); plt.axis('tight')    
    plt.show()
    
if __name__=="__main__":
    rm = Ising_Metropolis_MCMC(N=32)
    rm.simulate(1001,idxes=[1,4,32])
