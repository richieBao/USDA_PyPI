# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 19:36:16 2023

@author: richie bao
"""
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.colors as mcolors
import random
from pylab import mpl

def G_drawing(G,edge_labels=None,node_labels=None,routes=[],nodes=[],**kwargs):
    '''
    绘制复杂网络

    Parameters
    ----------
    G : networkx.classes.graph.Graph
        复杂网络（图）.
    edge_labels : string, optional
        显示边属性. The default is None.
    node_labels : string, optional
        显示节点属性. The default is None.
    routes : list(G vertex), optional
        构成图路径的顶点. The default is None.  
    nodes : list(G vertex), optional
        顶点的嵌套列表，用于不同顶点集的不同显示（颜色和大小等）. The default is None.        
    **kwargs : kwargs
        图表样式参数，包括options和sytle，默认值为：
            options={
                    "font_size": 20,
                    "font_color":"black",
                    "node_size": 150,
                    "node_color": "olive",
                    "edgecolors": "olive",
                    "linewidths": 7,
                    "width": 1,
                    "with_labels":True,    
                    }
             style={
                    "figsize":(3,3),   
                    "tight_layout":True,
                    "pos_func":nx.spring_layout,
                    "edge_label_font_size":10,
                    "pos":None
                    }.

    Returns
    -------
    None.

    '''    
    plt.rc('axes', unicode_minus=False) # 解决图表负号不正确显示问题
    mpl.rcParams['font.sans-serif']=['SimHei'] # 解决中文字符乱码问题

    def generate_color():
        color = '#{:02x}{:02x}{:02x}'.format(*map(lambda x: random.randint(0, 255), range(3)))
        return color
    
    options={
    "font_size": 20,
    "font_color":"black",
    "node_size": 150,
    "node_color": "olive",
    "edgecolors": "olive",
    "edge_color":'k',
    "linewidths": 7,
    "width": 1,
    "with_labels":True,    
    "cmap":None,
    }
    options.update((k, kwargs[k]) for k in set(kwargs).intersection(options))
    
    style={
    "figsize":(3,3),   
    "tight_layout":True,
    "pos_func":nx.spring_layout,
    "edge_label_font_size":10,
    "pos":None,
    "edge_colors":list(mcolors.TABLEAU_COLORS.values()),
    "edge_widths":[3]*len(routes),
    "title":None,
    "nodes_size":[200]*len(nodes),
    "nodes_color":[generate_color() for i in range(len(nodes))]#list(mcolors.TABLEAU_COLORS.values()),
    }
    
    style.update((k, kwargs[k]) for k in set(kwargs).intersection(style))        
    fig,ax=plt.subplots(figsize=style['figsize'],tight_layout=style["tight_layout"]) 
    
    if style['pos']:
        pos=style['pos']
    else:
        pos=list(map(style["pos_func"],[G]))[0]    
        
    if routes:
        route_edges=[[(r[n],r[n+1]) for n in range(len(r)-1)] for r in routes]
        [nx.draw_networkx_edges(G,pos=pos,edgelist=edgelist,edge_color=style['edge_colors'][idx],width=style['edge_widths'][idx],) for idx,edgelist in enumerate(route_edges)]        

    
    if node_labels:
        options["with_labels"]=False
        nx.draw(G, pos=pos,ax=ax,**options)
        node_labels=nx.get_node_attributes(G,node_labels)
        nx.draw_networkx_labels(G, pos, labels=node_labels,ax=ax)
    else:
        nx.draw(G, pos=pos,ax=ax,**options)        
    
    if edge_labels:
        edge_labels=nx.get_edge_attributes(G,edge_labels)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,ax=ax,font_size=style["edge_label_font_size"])  
        
    if nodes:
        [nx.draw_networkx_nodes(G,pos=pos,nodelist=sub_nodes,node_size=style['nodes_size'][idx],node_color=style['nodes_color'][idx]) for idx,sub_nodes in enumerate(nodes)]    
        
    plt.title(style['title'])
    plt.show()  

