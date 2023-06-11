# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 08:32:40 2023

@author: richie bao
"""
from pgmpy.models import FactorGraph
from pgmpy.factors.discrete import DiscreteFactor
import networkx as nx
import matplotlib.pyplot as plt

def draw_factor_graph(factor_graph,pos=None,gibbs_pos=None,show_pos=False,**kwargs):
    # nx.draw(nx_graph,with_labels=True)
    options = {
        "font_size": 20,
        "node_size": 900,
        "node_color": "gray",
        "edgecolors": "black",
        "linewidths": 1,
        "width": 1,
    }
    options.update((k, kwargs[k]) for k in set(kwargs).intersection(options))
    
    style={"figsize":(10, 3),
           "margins":0.1,
           "offset":0.05,
           "f_node_size":800,
           "f_node_color": 'white',
           "f_edgecolors": "black",
           "f_linewidths": 1,
           "text_fontsize":8}    # float or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
    style.update((k, kwargs[k]) for k in set(kwargs).intersection(style))   
    
    nx_graph=nx.Graph(factor_graph.edges)
    nodes=list(nx_graph.nodes)

    gibbs_mapping={}
    nodes_mapping={}

    phi_idx=1
    for node in nodes:
        if isinstance(node,DiscreteFactor):
            new_node_name=f'f{phi_idx}' #f'f-{node.state_names}'
            nodes_mapping[node]=new_node_name
            gibbs_mapping[new_node_name]=node
            phi_idx+=1
        else:
            nodes_mapping[node]=node        

    # nodes_mapping={i:f'f-{i.state_names}' if isinstance(i,DiscreteFactor) else i for i in nodes}
    fig, ax = plt.subplots(figsize=style["figsize"])
    nx_graph=nx.relabel_nodes(nx_graph, nodes_mapping)
    # print(nx_graph.nodes)
    
    if pos:
        pos=pos
    else:
        x,y=0,0
        pos={}
        for n in nx_graph.nodes:
            if n[0]=='f':
                pos[n]=(x,y-1)
            else:
                pos[n]=(x,y)
            x+=1
            
    if gibbs_pos:
        gibbs_pos=gibbs_pos
    else:    
        gibbs_pos={k:[[pos[k][0]+style["offset"],pos[k][1]+style["offset"]],v] for k,v in gibbs_mapping.items()}
    
    if show_pos:
        print(f"node pos:{pos};\ngibbs pos:{gibbs_pos}")    
    
    nx.draw_networkx(nx_graph, pos,**options)
    nx.draw_networkx_nodes(nx_graph, pos, 
                           node_size=style["f_node_size"], 
                           nodelist=gibbs_mapping.keys(),
                           node_shape='s',
                           node_color=style["f_node_color"], 
                           edgecolors=style["f_edgecolors"],
                           linewidths=style["f_linewidths"]) # matplotlib.markers
    for k,v in gibbs_pos.items():
        ax.text(v[0][0],v[0][1],v[1],fontsize=style["text_fontsize"])

    # Set margins for the axes so that nodes aren't clipped
    ax = plt.gca()
    ax.margins(style["margins"])
    plt.axis("off")
    plt.show()      
    
def draw_grid_graph(n_x,n_y,diagonal=False,**kwargs):
    options = {
        "font_size": 20,
        "node_size": 900,
        "node_color": "gray",
        "edgecolors": "black",
        "linewidths": 1,
        "width": 1,
    }    
    options.update((k, kwargs[k]) for k in set(kwargs).intersection(options))
    
    style={"figsize":(5, 5),
           "margins":0.1,
           "offset":0.05,
           "f_node_size":800,
           "f_node_color": 'white',
           "f_edgecolors": "black",
           "f_linewidths": 1,
           "text_fontsize":8}    # float or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
    style.update((k, kwargs[k]) for k in set(kwargs).intersection(style))    
    
    fig, ax = plt.subplots(figsize=style["figsize"])
    G = nx.grid_2d_graph(n_x, n_y) 
    
    if diagonal:
        nghs_diagonal=[]    
        for n in G.nodes:
            ngh_lt=(n[0]-1,n[1]+1)
            ngh_lb=(n[0]-1,n[1]-1)
            ngh_rt=(n[0]+1,n[1]+1)    
            ngh_rb=(n[0]+1,n[1]-1)
            if ngh_lt in G.nodes:
                nghs_diagonal.append((n,ngh_lt))
            if ngh_rt in G.nodes:
                nghs_diagonal.append((n,ngh_rt))    
            if ngh_lb in G.nodes:
                nghs_diagonal.append((n,ngh_lb))            
            if ngh_rb in G.nodes:
                nghs_diagonal.append((n,ngh_rb))            
        G.add_edges_from(nghs_diagonal)

    pos={n:p for n,p in zip(G.nodes,G.nodes)}
    nx.draw(G,pos=pos,**options)
    for node,pos in pos.items():
        ax.text(pos[0],pos[1],f'$X_{node}$')

    ax = plt.gca()
    ax.margins(style["margins"])
    plt.axis("off")
    plt.show()    
