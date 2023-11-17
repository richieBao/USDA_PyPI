# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 15:34:39 2023

@author: richie bao 
ref:Variational Autoencoders explained — with PyTorch Implementation, https://sannaperzon.medium.com/paper-summary-variational-autoencoders-with-pytorch-implementation-1b4b23b1763a
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
import torchvision.datasets as datasets  
from torch.utils.data import DataLoader  
import matplotlib.pyplot as plt


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, z_dim, h_dim=200):
        super().__init__()
        # encoder
        self.img_2hid = nn.Linear(input_dim, h_dim)

        # one for mu and one for stds, note how we only output
        # diagonal values of covariance matrix. Here we assume
        # the pixels are conditionally independent 
        self.hid_2mu = nn.Linear(h_dim, z_dim)
        self.hid_2sigma = nn.Linear(h_dim, z_dim)

        # decoder
        self.z_2hid = nn.Linear(z_dim, h_dim)
        self.hid_2img = nn.Linear(h_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.img_2hid(x))
        mu = self.hid_2mu(h)
        sigma = self.hid_2sigma(h)
        return mu, sigma

    def decode(self, z):
        new_h = F.relu(self.z_2hid(z))
        x = torch.sigmoid(self.hid_2img(new_h))
        return x

    def forward(self, x):
        mu, sigma = self.encode(x)

        # Sample from latent distribution from encoder
        epsilon = torch.randn_like(sigma)
        z_reparametrized = mu + sigma*epsilon

        x = self.decode(z_reparametrized)
        return x, mu, sigma
    
# Define train function
def vae_train(num_epochs, model, optimizer, loss_fn,train_loader,INPUT_DIM,device):
    # Start training
    for epoch in range(num_epochs):
        loop = tqdm(enumerate(train_loader))
        for i, (x, y) in loop:
            # Forward pass
            x = x.to(device).view(-1, INPUT_DIM)
            x_reconst, mu, sigma = model(x)

            # loss, formulas from https://www.youtube.com/watch?v=igP03FXZqgo&t=2182s
            reconst_loss = loss_fn(x_reconst, x)
            kl_div = - torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

            # Backprop and optimize
            loss = reconst_loss + kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

def vae_inference_(digit,model,dataset,device, num_examples=1):
    """
    Generates (num_examples) of a particular digit.
    Specifically we extract an example of each digit,
    then after we have the mu, sigma representation for
    each digit we can sample from that.

    After we sample we can run the decoder part of the VAE
    and generate examples.
    """
    images = []
    idx = 0
    for x, y in dataset:
        if y == idx:
            images.append(x.to(device))
            idx += 1
        if idx == 10:
            break

    encodings_digit = []
    for d in range(10):
        with torch.no_grad():
            mu, sigma = model.encode(images[d].view(1, 784))
        encodings_digit.append((mu, sigma))

    mu, sigma = encodings_digit[digit]
    for example in range(num_examples):
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        out = model.decode(z)
        out = out.view(-1, 1, 28, 28)
        save_image(out, f"C:/Users/richie/omen_richiebao/omen_temp/generated_{digit}_ex{example}.png")    
        
def vae_digit_inference(digit,model,dataset,device, num_examples=1,figsize=(9,9),fontsize=10):
    """
    Generates (num_examples) of a particular digit.
    Specifically we extract an example of each digit,
    then after we have the mu, sigma representation for
    each digit we can sample from that.

    After we sample we can run the decoder part of the VAE
    and generate examples.
    """
    images = []
    idx = 0
    for x, y in dataset:
        if y == idx:
            images.append(x.to(device))
            idx += 1
        if idx == 10:
            break

    encodings_digit = []
    for d in range(10):
        with torch.no_grad():
            mu, sigma = model.encode(images[d].view(1, 784))
        encodings_digit.append((mu, sigma))

    mu, sigma = encodings_digit[digit]
    generative_digits=[]
    for example in range(num_examples):
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        out = model.decode(z)
        out = out.view(-1, 28, 28)
        # save_image(out, f"C:/Users/richie/omen_richiebao/omen_temp/generated_{digit}_ex{example}.png")            
        generative_digits.append(out.cpu())
        
    generative_digits_stack=torch.stack(generative_digits,dim=0)
    img_grid = torchvision.utils.make_grid(generative_digits_stack, nrow=num_examples, normalize=True, pad_value=0.9)
    img_grid = img_grid.permute(1, 2, 0)
    
    plt.figure(figsize=figsize)
    plt.title(f"digit:{digit}",size=fontsize)
    plt.imshow(img_grid)
    plt.axis('off')
    plt.show()
    plt.close()        
        
    
if __name__=="__main__":
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    INPUT_DIM = 784
    Z_DIM = 20
    H_DIM = 200
    NUM_EPOCHS = 10
    BATCH_SIZE = 32
    LR_RATE = 3e-4        
    
    dataset = datasets.MNIST(root="C:/Users/richie/omen_richiebao/omen_temp/dataset/", train=True, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

    
    # Initialize model, optimizer, loss
    model = VariationalAutoEncoder(INPUT_DIM, Z_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)
    loss_fn = nn.BCELoss(reduction="sum")
    
    # Run training
    vae_train(NUM_EPOCHS, model, optimizer, loss_fn,train_loader)
    
    totensor=transforms.ToTensor()
    vae_inference(9,model,dataset)
