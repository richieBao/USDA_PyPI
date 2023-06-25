# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 19:36:39 2023

@author: richie bao
ref:https://www.kaggle.com/code/namanmanchanda/rnn-in-pytorch/notebook;  
    https://www.kaggle.com/code/kanncaa1/recurrent-neural-network-with-pytorch/notebook
    https://machinelearningmastery.com/text-generation-with-lstm-in-pytorch/
"""
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as data

import os
import matplotlib.pyplot as plt
import numpy as np

class RNN_LSTM_sequence(nn.Module):
    def __init__(self,input_size=1,hidden_size=50,out_size=1,selection='LSTM'):
        super().__init__()
        self.hidden_size=hidden_size
        self.selection=selection
        if selection=='LSTM':
            self.rnn_lstm=nn.LSTM(input_size,hidden_size)             
            self.hidden=(torch.zeros(1,1,hidden_size),torch.zeros(1,1,hidden_size))
        elif selection=='RNN':   
            self.rnn_lstm=nn.RNN(input_size,hidden_size)
            self.hidden=torch.zeros(1,1,hidden_size)
            
        self.linear=nn.Linear(hidden_size,out_size)    

    def forward(self,seq):
        model_out,self.hidden=self.rnn_lstm(seq.view(len(seq),1,-1),self.hidden)
        pred=self.linear(model_out.view(len(seq),-1))
        return pred[-1]

def RNN_LSTM_train_sequence(model,train_data,train_set,x,y,optimizer,criterion,window_size,test_size,epochs=10,future=40,plot=False):
    for i in range(epochs):
        for seq,y_train in train_data:
            optimizer.zero_grad()
            if model.selection=='LSTM':
                model.hidden=(torch.zeros(1,1,model.hidden_size),torch.zeros(1,1,model.hidden_size))
            elif model.selection=='RNN':
                model.hidden=torch.zeros(1,1,model.hidden_size)
            y_pred=model(seq)
            loss=criterion(y_pred,y_train)
            loss.backward()
            optimizer.step()

        print(f'Epoch {i} Loss:{loss.item()};',end=' ' )
        preds=train_set[-window_size:].tolist()
        for f in range(future):
            seq=torch.FloatTensor(preds[-window_size:])
            with torch.no_grad():
                if model.selection=='LSTM':
                    model.hidden=(torch.zeros(1,1,model.hidden_size),torch.zeros(1,1,model.hidden_size))
                elif model.selection=='RNN':
                    model.hidden=torch.zeros(1,1,model.hidden_size)
                preds.append(model(seq).item())
                
        loss = criterion(torch.tensor(preds[-window_size:]), y[-test_size:-test_size+window_size])
        print(f"Performance on test range: {loss}")
        if plot:
            fig, ax=plt.subplots(1, 1,figsize=(10,2))
            ax.plot(x[-test_size-window_size:],y[-test_size-window_size:], lw=3, alpha=0.6)
            ax.plot(x[-test_size:-test_size+window_size],preds[window_size:], lw=3, alpha=0.6,color='orange')
            ax.set_xlim([x[-test_size-window_size],x[-1]])
            ax.spines[['right', 'top']].set_visible(False)
            plt.show()
            
class RNN_model_img(nn.Module):
    def __init__(self,input_dim,hidden_dim,layer_dim,output_dim,nonlinearity='relu',batch_first=True):
        super(RNN_model_img,self).__init__()

        self.hidden_dim=hidden_dim
        self.layer_dim=layer_dim
        self.rnn=nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=batch_first, nonlinearity=nonlinearity)
        self.fc=nn.Linear(hidden_dim,output_dim)

    def forward(self,x):
        h0=Variable(torch.zeros(self.layer_dim,x.size(0),self.hidden_dim))
        out,hn=self.rnn(x,h0)
        out=self.fc(out[:,-1,:])
        return out

def RNN_train_img(model,train_loader,test_loader,input_dim,optimizer,error,epochs=10,step_eval=250,step_print=500):
    loss_list=[]
    iteration_list=[]
    accuracy_list=[]
    count=0    

    for epoch in range(epochs):
        for i,(imgs,labels) in enumerate(train_loader):        
            train=Variable(torch.squeeze(imgs))
            labels=Variable(labels)
            
            optimizer.zero_grad() # Clear gradients
            outputs=model(train) # Forward propagation
            loss=error(outputs, labels) # Calculate softmax and ross entropy loss
            loss.backward() # Calculating gradients
            optimizer.step() # Update parameters
            count+=1

            if count%step_eval==0:
                correct=0
                total=0
                for imgs,labels in test_loader:
                    test=Variable(torch.squeeze(imgs))
                    outputs=model(test)
                    predicted=torch.max(outputs.data,1)[1]
                    total+=labels.size(0)
                    correct+=(predicted==labels).sum()

                accuracy=100*correct/float(total)
                loss_list.append(loss.data)
                iteration_list.append(count)
                accuracy_list.append(accuracy)         

                if count%step_print==0:
                    print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data.numpy(), accuracy))

    return iteration_list,loss_list,accuracy_list        

class CharModel(nn.Module):
    def __init__(self,n_vocab,input_size=1, hidden_size=256, num_layers=2, batch_first=True, dropout_lstm=0.2,dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first, dropout=dropout_lstm)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, n_vocab)
    def forward(self, x):
        x, _ = self.lstm(x)
        # take only the last output
        x = x[:, -1, :]
        # produce output
        x = self.linear(self.dropout(x))
        return x

def char_train(model,loader,epochs,char_to_int,save_path=None):
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'device={device}')   
    model.to(device)    
    optimizer=optim.Adam(model.parameters())
    
    if os.path.exists(save_path):
        checkpoint= torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_trained = checkpoint['epoch']
        print('loaded checkpoint')
    else:
        epoch_trained=0    
    
    loss_func=nn.CrossEntropyLoss(reduction="sum")
        
    best_model=None
    best_loss=np.inf

    for epoch in range(epochs):
        model.train()
        for X_batch,y_batch in loader:
            y_pred=model(X_batch.to(device))
            loss=loss_func(y_pred,y_batch.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # validation
        model.eval()
        loss=0
        num=0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                y_pred = model(X_batch.to(device))
                loss += loss_func(y_pred, y_batch.to(device))
                num+=1

            if loss<best_loss:
                best_loss=loss
                best_model=model.state_dict()            

                if save_path:
                    torch.save({
                        'epoch':epoch+epoch_trained,
                        'model_state_dict':best_model,
                        'optimizer_state_dict':optimizer.state_dict(),
                        'loss':best_loss,
                        'char_to_int':char_to_int}, 
                        save_path)
                elif save_path and char_to_int:
                    torch.save([best_model, char_to_int], "single-char.pth")

            print("Epoch %d: Cross-entropy: %.4f" % (epoch+epoch_trained, loss/num))    
            
def char_random_generation(model,checkpoint_path,filename,seq_length=100,gen_length=1000):
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    checkpoint=torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    char_to_int=checkpoint['char_to_int']
    
    n_vocab = len(char_to_int)
    int_to_char = dict((i, c) for c, i in char_to_int.items())
    
    raw_text = open(filename, 'r', encoding='utf-8').read()
    raw_text = raw_text.lower()
    start = np.random.randint(0, len(raw_text)-seq_length)
    prompt = raw_text[start:start+seq_length]
    pattern = [char_to_int[c] for c in prompt]
    
    model.eval()
    print('Prompt: "%s"' % prompt,'\n','-'*50)    
    with torch.no_grad():
        for i in range(gen_length):
            # format input array of int into PyTorch tensor
            x = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
            x = torch.tensor(x, dtype=torch.float32)
            # generate logits as output from the model
            prediction = model(x.to(device))
            # convert logits into one character
            index = int(prediction.argmax())
            result = int_to_char[index]
            print(result, end="")
            # append the new character into the prompt for the next iteration
            pattern.append(index)
            pattern = pattern[1:] # -seq_length            