# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 15:50:08 2023

@author: richie bao
"""
import pandas as pd
from nltk.tokenize import word_tokenize
import numpy as np

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