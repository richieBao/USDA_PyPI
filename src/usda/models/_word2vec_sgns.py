# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 08:48:27 2023

@author: richie bao
ref: implementation of word2vec Paper https://www.kaggle.com/code/ashukr/implementation-of-word2vec-paper
"""
import torch
from torch import nn
import torch.optim as optim
import numpy as np
import random
import os

if __package__:
    from ._nlp_tools import get_batches4word2vec  
else:
    from _nlp_tools import get_batches4word2vec

class SkipGramNeg(nn.Module):
    def __init__(self, n_vocab, n_embed, noise_dist=None):
        super().__init__()
        
        self.n_vocab = n_vocab
        self.n_embed = n_embed
        self.noise_dist = noise_dist
        
        # define embedding layers for input and output words
        self.in_embed = nn.Embedding(n_vocab,n_embed)
        self.out_embed = nn.Embedding(n_vocab,n_embed)
        
        # Initialize both embedding tables with uniform distribution
        self.in_embed.weight.data.uniform_(-1,1)
        self.out_embed.weight.data.uniform_(-1,1)
        
    def forward_input(self, input_words):
        # return input vector embeddings
        input_vector = self.in_embed(input_words)
        return input_vector
    
    def forward_output(self, output_words):
        # return output vector embeddings
        output_vector = self.out_embed(output_words)

        return output_vector
    
    def forward_noise(self, batch_size, n_samples,device):
        """ Generate noise vectors with shape (batch_size, n_samples, n_embed)"""
        if self.noise_dist is None:
            # Sample words uniformly
            noise_dist = torch.ones(self.n_vocab)
        else:
            noise_dist = self.noise_dist
            
        # Sample words from our noise distribution
        noise_words = torch.multinomial(noise_dist,
                                        batch_size * n_samples,
                                        replacement=True)
        
        #device = "cuda" if model.out_embed.weight.is_cuda else "cpu"
        noise_words = noise_words.to(device)
        
        ## TODO: get the noise embeddings
        # reshape the embeddings so that they have dims (batch_size, n_samples, n_embed)
        # as we are adding the noise to the output, so we will create the noise vectr using the
        # output embedding layer
        noise_vector = self.out_embed(noise_words).view(batch_size,n_samples,self.n_embed)        
        return noise_vector

class NegativeSamplingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_vectors, output_vectors, noise_vectors):
        
        batch_size, embed_size = input_vectors.shape
        
        # Input vectors should be a batch of column vectors
        input_vectors = input_vectors.view(batch_size, embed_size, 1)
        
        # Output vectors should be a batch of row vectors
        output_vectors = output_vectors.view(batch_size, 1, embed_size)
        
        # bmm = batch matrix multiplication
        # correct log-sigmoid loss
        out_loss = torch.bmm(output_vectors, input_vectors).sigmoid().log()
        out_loss = out_loss.squeeze()
        
        #debugging
        #print(type(noise_vectors)) #it is a tensor
        
        #'neg' returns the negative of a tensor
        #print(noise_vectors)
        #print(noise_vectors.neg())
        
        # incorrect log-sigmoid loss
        noise_loss = torch.bmm(noise_vectors.neg(), input_vectors).sigmoid().log()
        noise_loss = noise_loss.squeeze().sum(1)  # sum the losses over the sample of noise vectors

        # negate and sum correct and noisy log-sigmoid losses
        # return average batch loss
        return -(out_loss + noise_loss).mean()
    
def cosine_similarity(embedding, valid_size=16, valid_window=100, device='cpu'):
    """ Returns the cosine similarity of validation words with words in the embedding matrix.
        Here, embedding should be a PyTorch embedding module.
    """
    
    # Here we're calculating the cosine similarity between some random words and 
    # our embedding vectors. With the similarities, we can look at what words are
    # close to our random words.
    
    # sim = (a . b) / |a||b|
    
    embed_vectors = embedding.weight
    
    # magnitude of embedding vectors, |b|
    magnitudes = embed_vectors.pow(2).sum(dim=1).sqrt().unsqueeze(0)
    
    # pick N words from our ranges (0,window) and (1000,1000+window). lower id implies more frequent 
    valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
    valid_examples = np.append(valid_examples,
                               random.sample(range(1000,1000+valid_window), valid_size//2))
    valid_examples = torch.LongTensor(valid_examples).to(device)
    
    valid_vectors = embedding(valid_examples)
    similarities = torch.mm(valid_vectors, embed_vectors.t())/magnitudes
        
    return valid_examples, similarities    

def noise_dist4sgns(freqs):
    word_freqs = np.array(sorted(freqs.values(), reverse=True))
    unigram_dist = word_freqs/word_freqs.sum()
    noise_dist = torch.from_numpy(unigram_dist**(0.75)/np.sum(unigram_dist**(0.75)))

    return noise_dist     
   
def sgns_train(model,criterion,optimizer,train_words,int_to_vocab,print_every = 1500,steps = 0,epochs = 5,n_samples=5, device='cpu',save_path=None):       
    # train for some number of epochs
    if os.path.exists(save_path):
        checkpoint= torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_trained = checkpoint['epoch']
        print('loaded checkpoint')
    else:
        epoch_trained=0         
    
    best_loss=np.inf
    for epoch in range(epochs):
        # get our input, target batches
        for input_words, target_words in get_batches4word2vec(train_words, 512):
            steps += 1
            inputs, targets = torch.LongTensor(input_words), torch.LongTensor(target_words)
            inputs, targets = inputs.to(device), targets.to(device)
            
            # input, outpt, and noise vectors
            input_vectors = model.forward_input(inputs)
            output_vectors = model.forward_output(targets)
            noise_vectors = model.forward_noise(inputs.shape[0], n_samples,device)
            
            # negative sampling loss
            loss = criterion(input_vectors, output_vectors, noise_vectors)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # loss stats            
            if steps % print_every == 0:
                print(f'Epoch:{epoch+1++epoch_trained}/{epochs++epoch_trained};Loss:{loss.item()}')
                valid_examples, valid_similarities = cosine_similarity(model.in_embed, device=device)
                _, closest_idxs = valid_similarities.topk(6)
                valid_examples, closest_idxs = valid_examples.to('cpu'), closest_idxs.to('cpu')
                for ii, valid_idx in enumerate(valid_examples):
                    closest_words = [int_to_vocab[idx.item()] for idx in closest_idxs[ii]][1:]
                    print(int_to_vocab[valid_idx.item()] + " | " + ', '.join(closest_words))
                print("...\n")    
                
            if loss<best_loss:
                best_loss=loss
                best_model=model.state_dict()  
                if save_path:
                    torch.save({
                        'epoch':epoch+epoch_trained,
                        'model_state_dict':best_model,
                        'optimizer_state_dict':optimizer.state_dict(),
                        'loss':best_loss},
                        save_path)              

if __name__=="__main__":
    import _nlp_tools as tools

    text_fn=r'I:\data\text8'
    with open(text_fn) as f:
        text=f.read()
    #print(text[:100])
    words=tools.text_replace_preprocess(text)
    print(words[:10])
    print("Total words in text: {}".format(len(words)))
    print("Unique words: {}".format(len(set(words)))) 
    
    vocab_to_int, int_to_vocab = tools.create_lookup_tables4vocab(words)
    int_words = [vocab_to_int[word] for word in words]
    print(int_words[:10])
    train_words,freqs=tools.subsampling_of_frequent_words(int_words)
    print(train_words[:10])
    
    batch_size=4
    X,y=next(tools.get_batches4word2vec(train_words,batch_size))
    print('-'*50)
    print(X,y)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    noise_dist =noise_dist4sgns(freqs)
    
    # instantiating the model
    embedding_dim=300
    model = SkipGramNeg(len(vocab_to_int), embedding_dim, noise_dist=noise_dist).to(device)  
    print(f'model:\n{model}')        
    
    # using the loss that we defined
    criterion = NegativeSamplingLoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    
    save_path=r'I:\model_ckpts\word2vec\sgns_model.pth'
    sgns_train(model,criterion,optimizer,train_words,int_to_vocab,device=device,save_path=save_path)
    
    
        
        




