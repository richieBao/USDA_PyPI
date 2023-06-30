# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 15:50:08 2023

@author: richie bao
"""
import pandas as pd
from nltk.tokenize import word_tokenize
import numpy as np
import re
from collections import Counter
import random
import os

import unicodedata

def build_co_occurrence_matrix(corpus,window_size):
    '''
    ref:co occurrence matrix algorithm, https://www.kaggle.com/code/tom99763/co-occurrence-matrix-algorithm
    '''        
    #build unique words
    unique_words=set()
    for text in corpus:
        for word in word_tokenize(text):
            unique_words.add(word)
  
    word_search_dict={word:np.zeros(shape=(len(unique_words))) for word in unique_words}
    word_list=list(word_search_dict.keys())
    for text in corpus:
        text_list=word_tokenize(text)
        for idx,word in enumerate(text_list):
            #pick word in the size range
            i=max(0,idx-window_size)
            j=min(len(text_list)-1,idx+window_size)
            search=[text_list[idx_] for idx_ in range(i,j+1)]
            search.remove(word)
            for neighbor in search:
                # get neighbor idx in word_search_dict
                nei_idx=word_list.index(neighbor)
                word_search_dict[word][nei_idx]+=1

    coo_df=pd.DataFrame(word_search_dict,index=word_search_dict.keys()).astype('int')
    return word_search_dict,coo_df

def text_replace_preprocess(text):
    '''
    transfered: word2vec utils: https://www.kaggle.com/datasets/ashukr/word2vec-utils
    '''

    # Replace punctuation with tokens so we can use them in our model
    text = text.lower()
    text = text.replace('.', ' <PERIOD> ')
    text = text.replace(',', ' <COMMA> ')
    text = text.replace('"', ' <QUOTATION_MARK> ')
    text = text.replace(';', ' <SEMICOLON> ')
    text = text.replace('!', ' <EXCLAMATION_MARK> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace('(', ' <LEFT_PAREN> ')
    text = text.replace(')', ' <RIGHT_PAREN> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    # text = text.replace('\n', ' <NEW_LINE> ')
    text = text.replace(':', ' <COLON> ')
    words = text.split()
    
    # Remove all words with  5 or fewer occurences
    word_counts = Counter(words)
    trimmed_words = [word for word in words if word_counts[word] > 5]

    return trimmed_words

def create_lookup_tables4vocab(words):
    """
    Create lookup tables for vocabulary
    :param words: Input list of words
    :return: Two dictionaries, vocab_to_int, int_to_vocab
    
    transfered: word2vec utils: https://www.kaggle.com/datasets/ashukr/word2vec-utils
    """
    word_counts = Counter(words)
    # sorting the words from most to least frequent in text occurrence
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    # create int_to_vocab dictionaries
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab

def subsampling_of_frequent_words(int_words,threshold=1e-5):
    '''
    transfered:implementation of word2vec Paper: https://www.kaggle.com/code/ashukr/implementation-of-word2vec-paper
    '''
    word_counts = Counter(int_words)
    total_count = len(int_words)
    freqs = {word: count/total_count for word, count in word_counts.items()}
    p_drop = {word: 1 - np.sqrt(threshold/freqs[word]) for word in word_counts}
    
    train_words = [word for word in int_words if random.random() < (1 - p_drop[word])]
    
    return train_words,freqs
    
def get_target4word2vec(words, idx, window_size=5):
    ''' Get a list of words in a window around an index. 
    transfered:implementation of word2vec Paper: https://www.kaggle.com/code/ashukr/implementation-of-word2vec-paper
    '''
    
    R = np.random.randint(1, window_size+1)
    start = idx - R if (idx - R) > 0 else 0
    stop = idx + R
    target_words = words[start:idx] + words[idx+1:stop+1]
    
    return list(target_words)

def get_batches4word2vec(words, batch_size, window_size=5):
    ''' Create a generator of word batches as a tuple (inputs, targets) 
    transfered:implementation of word2vec Paper: https://www.kaggle.com/code/ashukr/implementation-of-word2vec-paper
    '''
    
    n_batches = len(words)//batch_size
    
    # only full batches
    words = words[:n_batches*batch_size]
    
    for idx in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[idx:idx+batch_size]
        for ii in range(len(batch)):
            batch_x = batch[ii]
            batch_y = get_target4word2vec(batch, ii, window_size)
            y.extend(batch_y)
            x.extend([batch_x]*len(batch_y))
        yield x, y
        
#------------------------------------------------------------------------------         

class Word2idx_idx2word:
    '''
    ref: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html (NLP FROM SCRATCH: TRANSLATION WITH A SEQUENCE TO SEQUENCE NETWORK AND ATTENTION)
    '''
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
                       
# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()     

def normalizeString_cmn(s):
    s = s.lower().strip()
    if ' ' not in s:
        s = list(s)
        s = ' '.join(s)
    s = unicodeToAscii(s)  
    s = re.sub(r"([.。!！?？])", "", s)
    return s
    

def readLangs2langs(root,lang1, lang2, reverse=False,cmn=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open(os.path.join(root,'%s-%s.txt' % (lang1, lang2)), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    
    if cmn:
        pairs = [[normalizeString_cmn(s) for s in l.split('\t')[:2]] for l in lines]
    else:
        pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Word2idx_idx2word(lang2)
        output_lang = Word2idx_idx2word(lang1)
    else:
        input_lang = Word2idx_idx2word(lang1)
        output_lang = Word2idx_idx2word(lang2)

    return input_lang, output_lang, pairs    

def filterPair(p,MAX_LENGTH,eng_prefixes):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)

def filterPairs(pairs,MAX_LENGTH,eng_prefixes):
    return [pair for pair in pairs if filterPair(pair,MAX_LENGTH,eng_prefixes)]

def prepareData4seq2seq(root,lang1, lang2,eng_prefixes,MAX_LENGTH = 10, reverse=False,cmn=False):
    input_lang, output_lang, pairs = readLangs2langs(root,lang1, lang2, reverse,cmn)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs,MAX_LENGTH,eng_prefixes)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs
            
if __name__=="__main__":
    root=r'I:\data\NLP_dataset' 
    MAX_LENGTH = 10
    eng_prefixes = (
        "i am ", "i m ",
        "he is", "he s ",
        "she is", "she s ",
        "you are", "you re ",
        "we are", "we re ",
        "they are", "they re "
    )
    
    hidden_size = 128
    batch_size = 32    
    lang1, lang2='eng', 'cmn'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_lang, output_lang,pairs, train_dataloader = usda_models.get_dataloader4seq2seq(batch_size,root,lang1, lang2,eng_prefixes,MAX_LENGTH,device,reverse=True,cmn=True)    
        
        
    
    
    
    # eng_fra_root=r'I:\data\NLP_dataset\eng-fra' # \eng-fra.txt'
    # # eng_fra_root=r'I:\data\NLP_dataset'
    
    # MAX_LENGTH = 10
    
    # eng_prefixes = (
    #     "i am ", "i m ",
    #     "he is", "he s ",
    #     "she is", "she s ",
    #     "you are", "you re ",
    #     "we are", "we re ",
    #     "they are", "they re "
    # )
    # input_lang, output_lang, pairs =prepareData4seq2seq(eng_fra_root,'eng', 'fra',True) # 'eng', 'fra','eng','cmn' 
    # print(random.choice(pairs))

    # print(filterPairs([["i am ok","i m ok"]]))
    
    # print(normalizeString('apple is 22 delicious ~!'))
    
    # print(unicodeToAscii('\u0660'))
    
    # w2i=Word2idx_idx2word('delicious')
    # w2i.addWord('fruit')
    # w2i.addWord('fruit')
    # w2i.addWord('apple')
    # print(w2i.word2index,w2i.word2count,w2i.index2word)
    
    
    