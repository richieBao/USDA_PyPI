# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 09:21:53 2023

@author: richiebao
    
ref:
Huffman Coding using Priority Queue: https://www.geeksforgeeks.org/huffman-coding-using-priority-queue/
Huffman Code: https://allendowney.github.io/DSIRP/huffman.html
"""

import queue
import networkx as nx
import EoN
from EoN import hierarchy_pos
import matplotlib.pyplot as plt
import copy
 
# Maximum Height of Huffman Tree.
MAX_SIZE = 100
 
class HuffmanTreeNode:
    def __init__(self, character, frequency):
        # Stores character
        self.data = character
 
        # Stores frequency of the character
        self.freq = frequency
 
        # Left child of the current node
        self.left = None
 
        # Right child of the current node
        self.right = None
     
    def __lt__(self, other):
        return self.freq < other.freq
 
# Custom comparator class
class Compare:
    def __call__(self, a, b):
        # Defining priority on the basis of frequency
        return a.freq > b.freq
 
# Function to generate Huffman Encoding Tree
def generateTree(pq):
    # We keep on looping till only one node remains in the Priority Queue
    while pq.qsize() != 1:
        # Node which has least frequency
        left = pq.get()
 
        # Node which has least frequency
        right = pq.get()
 
        # A new node is formed with frequency left.freq + right.freq
        # We take data as '$' because we are only concerned with the frequency
        node = HuffmanTreeNode('$', left.freq + right.freq)
        node.left = left
        node.right = right
 
        # Push back node created to the Priority Queue
        pq.put(node)
        # print(node.__dict__)
    # print(pq.get().__dict__)
    return pq.get()
 
# Function to print the huffman code for each character.
# It uses arr to store the codes
huffman_encoding_dict={}
def printCodes(root, arr, top,verbose=True):
    # print('+',verbose)
    # Assign 0 to the left node and recur    
    if root.left:
        arr[top] = 0
        printCodes(root.left, arr, top + 1,verbose)
 
    # Assign 1 to the right node and recur
    if root.right:
        arr[top] = 1
        printCodes(root.right, arr, top + 1,verbose)
 
    # If this is a leaf node, then we print root.data
    # We also print the code for this character from arr
    if not root.left and not root.right:
        if verbose:
            print(root.data, end=' ')
            for i in range(top):
                print(arr[i], end='')
            print()
        huffman_encoding_dict[root.data]=arr[:top]

def HuffmanCodes(data, freq, size,verbose=True):
    # Declaring priority queue using custom comparator
    pq = queue.PriorityQueue()
 
    # Populating the priority queue
    for i in range(size):
        newNode = HuffmanTreeNode(data[i], freq[i])
        # print(newNode.__dict__)
        pq.put(newNode)
        
    # print(pq.get().__dict__)
    # print(pq.get().__dict__)
    # print(pq.get().__dict__)
    # print(pq.get().__dict__)
    
    # Generate Huffman Encoding Tree and get the root node
    root = generateTree(pq)
    # print(root.__dict__)
    # Print Huffman Codes
    arr = [0] * MAX_SIZE
    # print(arr)
    top = 0
    printCodes(root, arr, top,verbose)
    return root

def add_edges(parent, G):
    """Make a NetworkX graph that represents the tree."""
    if parent is None:
        return
    
    for child in (parent.left, parent.right):
        if child:
            G.add_edge(parent, child)
            add_edges(child, G)
def get_labels(parent, labels,decimal=3):
    if parent is None:
        return
    
    if parent.data == '$':
        labels[parent] = round(parent.freq,decimal)
    else:
        labels[parent] = parent.data
        
    get_labels(parent.left, labels)
    get_labels(parent.right, labels)
    
def get_edge_labels(parent, edge_labels):
    if parent is None:
        return
    
    if parent.left:
        edge_labels[parent, parent.left] = '0'
        get_edge_labels(parent.left, edge_labels)
        
    if parent.right:
        edge_labels[parent, parent.right] = '1'
        get_edge_labels(parent.right, edge_labels)    
    
def draw_tree(root,decimal=3,figsize=(5,5)):
    G = nx.DiGraph()
    add_edges(root, G)
    # print(G.edges)
    pos = hierarchy_pos(G)
    # print(pos)
    labels = {}
    get_labels(root, labels,decimal)
    # print(labels)
    edge_labels = {}
    get_edge_labels(root, edge_labels)
    # print(edge_labels)
    
    fig,ax=plt.subplots(figsize=figsize)
    nx.draw(G, pos, labels=labels, alpha=0.4,ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='C1',ax=ax)
    
flatten_lst=lambda lst: [m for n_lst in lst for m in flatten_lst(n_lst)] if type(lst) is list else [lst]

def huffman_encode(s,huffman_encoding_dict):
    t=flatten_lst([huffman_encoding_dict[letter] for letter in s])
    return ''.join(map(str,t))

def huffman_decode(code,root):
    s=[]
    root_bak=copy.deepcopy(root)
    for i in code:        
        if i=='0':
            child=root_bak.left
        elif i=='1':
            child=root_bak.right
        if child.left or child.right:  
            # print('+',i)
            # s.append(child.data)
            root_bak=child
        else:            
            s.append(child.data)
            # print('-',i)
            root_bak=copy.deepcopy(root)
    return ''.join(s)       

# Driver Code
if __name__ == '__main__':
    data = ['a',  'b', 'c', 'd', 'e']
    freq = [0.10, 0.15, 0.30,0.16, 0.29	]
    size = len(data)
 
    root=HuffmanCodes(data, freq, size,verbose=True)
    # draw_tree(root)
    
    # s='badbed'
    # s_encoded=huffman_encode(s,huffman_encoding_dict)
    # print(s_encoded)
    # s_decoded=huffman_decode(s_encoded,root)
    # print(s_decoded)