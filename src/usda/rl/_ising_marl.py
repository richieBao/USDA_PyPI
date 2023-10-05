# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 20:36:50 2023

@author: richie bao 
ref:Deep Reinforcement Learning In Action, https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction
Zai, A., & Brown, B. (2020). Deep reinforcement learning in action. Manning Publications.
"""
import numpy as np
import torch
from matplotlib import pyplot as plt

from collections import deque 
from random import shuffle 

class Ising_MARL:
    def __init__(self,size=(10,),hid_layer=20,epochs = 200,lr = 0.001 ):
        self.size=size
        self.hid_layer=hid_layer
        self.epochs=epochs
        self.lr=lr
        
        self.grid=self.init_grid()        
        self.grid_ = self.grid.clone() 
        self.grid__ = self.grid.clone()
        
        if len(size)==1:
            self.params=self.gen_params(size[0],4*hid_layer+hid_layer*2)  
        elif len(size)==2:
            self.params = self.gen_params(1,2*hid_layer+hid_layer*2) 
     
    def init_grid(self):
        grid = torch.randn(*self.size)
        grid[grid > 0] = 1
        grid[grid <= 0] = 0
        grid = grid.byte() 
        
        return grid
    
    def gen_params(self,N,size): 
        ret = []
        for i in range(N):
            vec = torch.randn(size) / 10.
            vec.requires_grad = True
            ret.append(vec)
        return ret
        
    def get_reward_1d(self,s,a): 
        r = -1
        for i in s:
            if i == a:
                r += 0.9
        r *= 2.
        return r    
    
    def qfunc(self,s,theta,layers=[(4,20),(20,2)],afn=torch.tanh):
        l1n = layers[0] 
        l1s = np.prod(l1n) 
        theta_1 = theta[0:l1s].reshape(l1n) 
        l2n = layers[1]
        l2s = np.prod(l2n)
        theta_2 = theta[l1s:l2s+l1s].reshape(l2n)
        bias = torch.ones((1,theta_1.shape[1]))
        l1 = s @ theta_1 + bias 
        l1 = torch.nn.functional.elu(l1)
        l2 = afn(l1 @ theta_2) 
        return l2.flatten()    
    
    def get_substate(self,b): 
        s = torch.zeros(2) 
        if b > 0: 
            s[1] = 1
        else:
            s[0] = 1
        return s
    
    def joint_state(self,s): 
        s1_ = self.get_substate(s[0]) 
        s2_ = self.get_substate(s[1])
        ret = (s1_.reshape(2,1) @ s2_.reshape(1,2)).flatten()  # array([1, 0, 0, 0]),array([0, 1, 0, 0]),array([0, 0, 1, 0]),array([0, 0, 0, 1])
        return ret   
    
    def softmax_policy(self,qvals,temp=0.9): 
        soft = torch.exp(qvals/temp) / torch.sum(torch.exp(qvals/temp))     
        action = torch.multinomial(soft,1) 
        return action    
    
    def get_coords(self,grid,j): 
        x = int(np.floor(j / grid.shape[0])) 
        y = int(j - x * grid.shape[0]) 
        return x,y
    
    def get_reward_2d(self,action,action_mean): 
        r = (action*(action_mean-action/2)).sum()/action.sum() 
        return torch.tanh(5 * r)     
    
    def mean_action(self,grid,j):
        x,y = self.get_coords(grid,j) 
        action_mean = torch.zeros(2) 
        for i in [-1,0,1]: 
            for k in [-1,0,1]:
                if i == k == 0:
                    continue
                x_,y_ = x + i, y + k
                x_ = x_ if x_ >= 0 else grid.shape[0] - 1
                y_ = y_ if y_ >= 0 else grid.shape[1] - 1
                x_ = x_ if x_ <  grid.shape[0] else 0
                y_ = y_ if y_ < grid.shape[1] else 0
                cur_n = grid[x_,y_]
                s = self.get_substate(cur_n) 
                action_mean += s
        action_mean /= action_mean.sum() 
        return action_mean    
            
    def train_1d(self):
        self.losses = [[] for i in range(self.size[0])] 
          
        for i in range(self.epochs):
            for j in range(self.size[0]): 
                l = j - 1 if j - 1 >= 0 else self.size[0]-1 
                r = j + 1 if j + 1 < self.size[0] else 0 
                state_ = self.grid[[l,r]] 
                state = self.joint_state(state_) 
                qvals = self.qfunc(state.float().detach(),self.params[j],layers=[(4,self.hid_layer),(self.hid_layer,2)])
                qmax = torch.argmax(qvals,dim=0).detach().item() 
                action = int(qmax)
                self.grid_[j] = action 
                reward = self.get_reward_1d(state_.detach(),action)
                with torch.no_grad(): 
                    target = qvals.clone()
                    target[action] = reward
                loss = torch.sum(torch.pow(qvals - target,2))
                self.losses[j].append(loss.detach().numpy())
                loss.backward()
                with torch.no_grad(): 
                    self.params[j] = self.params[j] - self.lr * self.params[j].grad
                self.params[j].requires_grad = True
        
            with torch.no_grad(): 
                self.grid.data =self.grid_.data  
                
    def train_2d(self,num_iter = 3,replay_size = 50,batch_size = 10,gamma = 0.9,temp=0.5):
        J = np.prod(self.size) 
        self.losses = [[] for i in range(J)]        
        layers = [(2,self.hid_layer),(self.hid_layer,2)]
        replay = deque(maxlen=replay_size) 
        
        for i in range(self.epochs): 
            act_means = torch.zeros((J,2)) 
            q_next = torch.zeros(J) 
            for m in range(num_iter): 
                for j in range(J): 
                    action_mean = self.mean_action(self.grid_,j).detach()
                    act_means[j] = action_mean.clone()
                    qvals = self.qfunc(action_mean.detach(),self.params[0],layers=layers)
                    action = self.softmax_policy(qvals.detach(),temp=temp)
                    self.grid__[self.get_coords(self.grid_,j)] = action
                    q_next[j] = torch.max(qvals).detach()
                self.grid_.data = self.grid__.data
            self.grid.data = self.grid_.data
            actions = torch.stack([self.get_substate(a.item()) for a in self.grid.flatten()])
            rewards = torch.stack([self.get_reward_2d(actions[j],act_means[j]) for j in range(J)])
            exp = (actions,rewards,act_means,q_next) 
            replay.append(exp)
            shuffle(replay)
            if len(replay) > batch_size: 
                ids = np.random.randint(low=0,high=len(replay),size=batch_size) 
                exps = [replay[idx] for idx in ids]
                for j in range(J):
                    jacts = torch.stack([ex[0][j] for ex in exps]).detach()
                    jrewards = torch.stack([ex[1][j] for ex in exps]).detach()
                    jmeans = torch.stack([ex[2][j] for ex in exps]).detach()
                    vs = torch.stack([ex[3][j] for ex in exps]).detach()
                    qvals = torch.stack([ self.qfunc(jmeans[h].detach(),self.params[0],layers=layers) \
                                         for h in range(batch_size)])
                    target = qvals.clone().detach()
                    target[:,torch.argmax(jacts,dim=1)] = jrewards + gamma * vs
                    loss = torch.sum(torch.pow(qvals - target.detach(),2))
                    self.losses[j].append(loss.item())
                    loss.backward()
                    with torch.no_grad():
                        self.params[0] = self.params[0] - self.lr * self.params[0].grad
                    self.params[0].requires_grad = True       
                           
def ising_1d_plot(losses,grid,size,figsize=(10,5)):
    fig,axes = plt.subplots(2,1,figsize=figsize)
    for i in range(size[0]):
        axes[0].scatter(np.arange(len(losses[i])),losses[i],s=3)
    print(grid,grid.sum())
    axes[1].imshow(np.expand_dims(grid,0))     
    
def ising_2d_plot(losses,grid,figsize=(10,5)):
    fig,ax = plt.subplots(2,1,figsize=figsize)
    fig.set_size_inches(5,3)
    ax[0].plot(np.array(losses).mean(axis=0))
    ax[1].imshow(grid)
    plt.tight_layout()
    plt.show()

    
if __name__=="__main__":
    #%% --------------------------------------------------------------------------
    size=(20,)
    hid_layer = 20 
    epochs = 3
    lr = 0.001 
    
    ising_1d=Ising_MARL(size,hid_layer,epochs,lr)
    print(ising_1d.grid)
    plt.imshow(np.expand_dims(ising_1d.grid,0))
    # print(ising_1d.params)
    ising_1d.train_1d()
    ising_1d_plot(ising_1d.losses,ising_1d.grid)
    #%% --------------------------------------------------------------------------
    # size=(10,10)
    # hid_layer = 10 
    # epochs = 200
    # lr = 0.0001     
    # num_iter = 3
    
    # ising_2d=Ising_MARL(size,hid_layer,epochs,lr)    
    # print(ising_2d.grid)
    # ising_2d.train_2d(num_iter)
    # ising_2d_plot(ising_2d.losses,ising_2d.grid)
