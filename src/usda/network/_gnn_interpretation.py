# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 07:55:06 2023

@author: richie bao
"""
from dash import Dash, dcc, html, Input, Output, callback, ctx, no_update
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import networkx as nx
from dash.exceptions import PreventUpdate
import torch
from torch.nn import Linear,Sequential, BatchNorm1d, ReLU, Dropout
from torch import nn
import numpy as np
import random
import copy

from torch_geometric.nn import GCNConv
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv,GATConv,GINConv
from torch.nn import LSTM
from torch_geometric.utils import to_dense_batch

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

#%%
class Random_Graph:
    def __init__(self,n_range=[5,12],p=0.3):
        self.n_range=n_range    
        self.p=p
        self.random_G=self.data_networkx()
        self.random_G_cytoscape=nx.cytoscape_data(self.random_G) 

    def data_networkx(self):
        n=random.randint(*self.n_range)
        gnp_random_G=nx.fast_gnp_random_graph(n, self.p)
        attrs={node:{'feature':random.randint(-5, 5)} for node in gnp_random_G.nodes()}
        attrs={k:{
            'label':'{}({})'.format(chr(97+k).upper(),v['feature']),
            'feature':v['feature'],
            'id':'{}'.format(chr(97+k).upper()),
            'value':'{}'.format(chr(97+k).upper()),
            } for k,v in attrs.items()}
        # print(attrs)
        nx.set_node_attributes(gnp_random_G, attrs)
        mapping={node:gnp_random_G.nodes[node]['id'] for node in gnp_random_G.nodes()}
        gnp_random_G=nx.relabel_nodes(gnp_random_G, mapping)
        
        return gnp_random_G

# rnd_G=Random_Graph()
# gnp_random_G=rnd_G.random_G
# random_G_cytoscape=rnd_G.random_G_cytoscape
#%%
flatten_lst=lambda lst: [m for n_lst in lst for m in flatten_lst(n_lst)] if type(lst) is list else [lst]
class GraphConstructor:
    @classmethod
    def data(cls):
        nodes = [
            {
                'data': {'id': short, 'label': f'{label}({f_1})','value':short,'feature':f_1},
                'position': {'x': lat, 'y': long},   
                'selected':True,
            } if short=='A' else
            {
                'data': {'id': short, 'label': f'{label}({f_1})','value':short,'feature':f_1},
                'position': {'x': lat, 'y': long},   
                'selected':False,
            }       
            for short, label, f_1,long, lat in (
                ('A', 'A', 6,1, 1),
                ('B', 'B', 2,2, 2),
                ('C', 'C', -10,1, 2),
                ('D', 'D', 1,3, 4),
                ('E', 'E', 3,0, 2),
            )
        ]        
        edges = [
            {'data': {'source': source, 'target': target}}
            for source, target in (
                ('A', 'C'),
                ('A', 'E'),
                ('E', 'B'),
                ('E', 'D'),
            )
        ]        
        # elements = nodes + edges    
        elements = {'nodes':nodes,'edges':edges}        
        return elements
    
    @classmethod
    def stylesheet(cls):
        stylesheet=[
                # Group selectors
                {
                    'selector': 'node',
                    'style': {
                        'content': 'data(label)',
                        # 'text-halign':'center',
                        # 'text-valign':'center',
                        # 'width':'label',
                        # 'height':'label',
                        'shape':'sphere'
                    }
                },]
        return stylesheet       
    
    @classmethod
    def data_networkx(cls):
        data_dict={
            'data': [],
            'directed': False,
            'multigraph': False,
            'elements': GraphConstructor.data()
            }
        G=nx.cytoscape_graph(data_dict)
        return G       

app.layout = dbc.Container([
    html.H4('GNN Algorithms'),
    dbc.Tabs([
        dbc.Tab(label='Vanilla', tab_id='tab-vanilla'),
        dbc.Tab(label='GCN', tab_id='tab-gcn'),
        dbc.Tab(label='GAT', tab_id='tab-gat'),
        dbc.Tab(label='GraphSAGE', tab_id='tab-graphsage'),
        dbc.Tab(label='GIN', tab_id='tab-gin'),
    ],id="tabs-gnns", active_tab='tab-vanilla', ),
    html.Div(id='tabs-content-graph')
])

@callback(Output('tabs-content-graph', 'children'),
              Input('tabs-gnns', 'active_tab'))
def render_content(tab):
    if tab == 'tab-vanilla':        
        return tab_vanilla()    
    elif tab == 'tab-gcn':
        return tab_gcn()  
    elif tab == 'tab-gat':
        return tab_gat()     
    elif tab == 'tab-graphsage':
        return tab_sage()     
    elif tab == 'tab-gin':
        return tab_gin()       
    
class Tab_general:
    @classmethod
    def tab_buttons(self):
        return html.Div([
            dbc.Button('Reset', id='button-reset',color="dark",n_clicks=0,className="me-1"),
            dbc.Button('Undo Last Update', id='button-undo',color="dark",n_clicks=0,className="me-1",disabled=True),
            dbc.Button('Update All Nodes', id='button-update',color="dark",n_clicks=0,className="me-1"),
            dbc.Button('Randomize Graph', id='button-randomize',color="dark",n_clicks=0,className="me-1"),
            ],
            # align="center",            
            )

#%% vGNN
G_fixed=GraphConstructor.data_networkx()
update_stack={'elements':[],'attrs':[]}
data_GC=GraphConstructor.data()
original_G=GraphConstructor.data_networkx()  

def tab_vanilla():      
    tab_div=dbc.Container([
        html.Br(),
        Tab_general.tab_buttons(),
        dbc.Row([
            dbc.Col([
                html.P(id='weight_iter'),       
                dcc.Slider(-10, 10, 0.1, value=1, marks=None,tooltip={"placement": "bottom", "always_visible": True},id='weight_1'),
                ],),                             
            dbc.Col([
                html.B('Initial Graph'),     
                cyto.Cytoscape(
                        id='graph_fixed',
                        elements=GraphConstructor.data(),
                        style={'width': '100%', 'height': '500px'},
                        layout={
                            'name': 'cose'
                        },
                        stylesheet=GraphConstructor.stylesheet(),
                    ),       
                ],),        
            ], 
            align="center",
            ),     
        dbc.Row([
            dbc.Col([
            html.P(id='current_node_id'),
            html.Div(id='formula_update'),
            html.P(dcc.Markdown('''
                                ---
                                
                                Here, $f$ is just ReLu: $f(x)=max(x,0)$,  
                                
                                Note that the weight $W^{(k)}$ is shared across all nodes!         
                                ''',mathjax=True)),
                ]),    
            # dbc.Col([
            #     html.P([
            #         ]),
            #     ]),            
            ], 
            align="center",
            ),    
           dbc.Row([          
               html.Div(id='cytoscape-tapNodeData-output'),
               ]),
    ],
    fluid=True,
    )    
    return tab_div      


def formula_update_content(G_fixed,data,weight_1,iter_num):
    neighbors=list(G_fixed.neighbors(data['id']))
    text=flatten_lst([['W',html.Sup(f'({iter_num+1})'),'h',html.Sub(f'({i})'),html.Sup(f'({iter_num})'),'+'] for i in [data['id']]+neighbors])[:-1]
    
    text_1=flatten_lst([[f'{weight_1}×','{:.3f}'.format(G_fixed.nodes[i]['feature']),'+'] for i in [data['id']]+neighbors])[:-1]
    result=round(sum([weight_1*G_fixed.nodes[i]['feature'] for i in [data['id']]+neighbors]),3)
    
    content=html.Div([
        html.Span(['h',html.Sup(f'({iter_num+1})'),html.Sub(data['id']),'=','f(']+text+[')']),
        html.Br(),
        html.Span(['=f(']+text_1+[')']),
        html.Br(),
        html.Span(['=f(']+[result]+[')']),
        html.Br(),
        html.Span([f'ReLU({result})=']+[max(result,0)]),
        ],
        style={'font-size':'24px'})    
    
    return content

def iter_info_update(current_node_id,iter_num):
        iter_info=[
            html.B(f'Next Update (Iteration {iter_num}):'),
            html.Br(),
            html.Span(f'Equation for Node {current_node_id}:')
            ]
        
        weight_iter_info=[
            html.B('Parameters for Next Update'),
            html.Br(),
            html.Span(['W',html.Sup(f'({iter_num})')])
            ] 
        return iter_info,weight_iter_info   

@callback(
    Output('formula_update', 'children'),
    Output('current_node_id', 'children'),
    Output('weight_iter', 'children'),
    Output('button-update', "n_clicks"),
    Output('button-undo', "n_clicks"),    
    Input('graph_fixed', 'tapNodeData'),
    Input('weight_1','value'),
    Input('button-update', "n_clicks"),
    Input('button-reset','n_clicks'),
    Input('button-undo', "n_clicks"),    
    )
def formula_update(data,weight_1,click_update,_,click_undo):    
    global G_fixed    
    global original_G
            
    if ctx.triggered_id == "button-reset":  
        # G_fixed=GraphConstructor.data_networkx()
        G_fixed=original_G
        data=G_fixed.nodes['A']     
        
        current_node_id=data['id']
        iter_num=1
        iter_info,weight_iter_info=iter_info_update(current_node_id,iter_num)
        return no_update,iter_info,weight_iter_info,0,0
    
    iter_num=abs(click_update-click_undo)+1
            
    if data is None:        
        data=G_fixed.nodes['A']           

    if data:
        content=formula_update_content(G_fixed,data,weight_1,iter_num-1)        
        current_node_id=data['id']
        iter_info,weight_iter_info=iter_info_update(current_node_id,iter_num)
        
        return content,iter_info,weight_iter_info,no_update,no_update      
   
     
class VanillaGNNLayer(torch.nn.Module):
    def __init__(self, dim_in, dim_out,weight=None):
        super().__init__()
        self.linear = Linear(dim_in, dim_out, bias=False)
        if weight is not None:
            self.linear.weight.data=torch.full((dim_in,dim_out),weight)

    def forward(self, x, adjacency):
        x = self.linear(x)
        x = torch.sparse.mm(adjacency, x)
        x=torch.relu(x)
        return x

@callback(
    Output('graph_fixed', 'elements'),
    Output('button-undo', 'disabled'),
    Input('button-update', "n_clicks"),
    Input('button-reset','n_clicks'),
    Input('button-undo', "n_clicks"),         
    Input('button-randomize', "n_clicks"),  
    Input('weight_1','value'),
    )
def update_all_nodes(click_update,_,click_undo,click_randomize,weight): 
    global update_stack
    global G_fixed
    global original_G
    global data_GC   

    if ctx.triggered_id == "button-randomize":  
        rnd_G=Random_Graph()
        G_fixed=rnd_G.random_G
        data_GC=rnd_G.random_G_cytoscape['elements']     
        original_G=copy.deepcopy(G_fixed)
        return data_GC,True      
    
    # print('---',click_update,click_undo)
    if ctx.triggered_id == "button-reset":  
        # G_fixed=original_G        
        return data_GC,True   
    
    if click_update is None:
        raise PreventUpdate
    elif click_update==0:
        return data_GC,True      
    
    if ctx.triggered_id == "button-update":                     
        net=VanillaGNNLayer(1,1,float(weight))
        feature=nx.get_node_attributes(G_fixed,'feature')
        X=torch.tensor(list(feature.values())).reshape(-1,1).to(torch.float)
        adj=nx.adjacency_matrix(G_fixed)
        adj.setdiag(1)
        adj=adj.todense()
        adj=torch.tensor(adj).to(torch.float)
        y=net(X,adj).detach().numpy().reshape(-1)   
        attrs_feature={k:{'feature':v} for k,v in zip(list(G_fixed.nodes()),y)}        
        attrs_label={k: {'label':f'{k}({v:.3f})'} for k,v in zip(list(G_fixed.nodes()),y)}
        
        nx.set_node_attributes(G_fixed, attrs_feature)
        nx.set_node_attributes(G_fixed, attrs_label)
        
        G_fixed_cytoscape=nx.cytoscape_data(G_fixed) 
        update_stack['elements'].append(G_fixed_cytoscape['elements'])       
        update_stack['attrs'].append([attrs_feature,attrs_label])
        return G_fixed_cytoscape['elements'],False   
        
    if ctx.triggered_id == "button-undo": 
        if len(update_stack['elements'])>1:
            attrs_feature,attrs_label=update_stack['attrs'].pop()
            nx.set_node_attributes(G_fixed, attrs_feature)
            nx.set_node_attributes(G_fixed, attrs_label)
            return update_stack['elements'].pop(),False     
        elif click_update-click_undo<2:
            G_fixed=GraphConstructor.data_networkx() 
            return data_GC,True
        else:            
            G_fixed=GraphConstructor.data_networkx() 
            return data_GC,True        
    
#%% tab_GCN 
gcn_G_fixed=GraphConstructor.data_networkx()
gcn_update_stack={'elements':[],'attrs':[]}
gcn_data_GC=GraphConstructor.data()
gcn_original_G=GraphConstructor.data_networkx()  

def tab_gcn():      
    tab_div=dbc.Container([
        html.Br(),
        html.Div([
            dbc.Button('Reset', id='gcn_button-reset',color="dark",n_clicks=0,className="me-1"),
            dbc.Button('Undo Last Update', id='gcn_button-undo',color="dark",n_clicks=0,className="me-1",disabled=True),
            dbc.Button('Update All Nodes', id='gcn_button-update',color="dark",n_clicks=0,className="me-1"),
            dbc.Button('Randomize Graph', id='gcn_button-randomize',color="dark",n_clicks=0,className="me-1"),
            ],
            # align="center",            
            ),
        dbc.Row([
            dbc.Col([
                html.P(id='gcn_weight_1_iter'),       
                dcc.Slider(-10, 10, 0.1, value=1, marks=None,tooltip={"placement": "bottom", "always_visible": True},id='gcn_weight_1'),
                # html.P(id='gcn_weight_2_iter'), 
                # dcc.Slider(-10, 10, 0.1, value=1, marks=None,tooltip={"placement": "bottom", "always_visible": True},id='gcn_weight_2'),                
                ],),       
            dbc.Col([
                html.B('Initial Graph'),     
                cyto.Cytoscape(
                        id='gcn_graph_fixed',
                        elements=GraphConstructor.data(),
                        style={'width': '100%', 'height': '500px'},
                        layout={
                            'name': 'cose'
                        },
                        stylesheet=GraphConstructor.stylesheet(),
                    ),       
                ],),        
            ], 
            align="center",
            ),     
        dbc.Row([
            dbc.Col([
            html.P(id='gcn_current_node_id'),
            html.Div(id='gcn_formula_update'),
            html.P(dcc.Markdown('''
                                ---
                                
                                Here, $f$ is just ReLu: $f(x)=max(x,0)$,  
                                
                                Note that the weight $W^{(k)}$ is shared across all nodes!         
                                ''',mathjax=True)),
                ]),    
            # dbc.Col([
            #     html.P([
            #         ]),
            #     ]),            
            ], 
            align="center",
            ),    
           dbc.Row([          
               html.Div(id='gcn_cytoscape-tapNodeData-output'),
               ]),
    ],
    fluid=True,
    )    
    return tab_div  

def gcn_iter_info_update(current_node_id,iter_num):
        iter_info=[
            html.B(f'Next Update (Iteration {iter_num}):'),
            html.Br(),
            html.Span(f'Equation for Node {current_node_id}:')
            ]
        
        weight_1_iter_info=[
            html.B('Parameters for Next Update'),
            html.Br(),
            html.Span(['W',html.Sup(f'({iter_num})')])
            ] 
        
        # weight_2_iter_info=[
        #     html.Span(['B',html.Sup(f'({iter_num})')])
        #     ]
        
        return iter_info,weight_1_iter_info 
    
def gcn_formula_update_content(gcn_G_fixed,data,weight_1,iter_num):
    current_node_id=data['id']
    neighbors=list(gcn_G_fixed.neighbors(current_node_id))
    # text_1=flatten_lst(['W',html.Sup(f'({iter_num+1})'),' x ','(',
    #       flatten_lst([['h',html.Sub(f'({i})'),html.Sup(f'({iter_num})'),'+'] for i in neighbors])[:-1],')','/',str(len(neighbors)),
    #       '+','B',html.Sup(f'({iter_num+1})'),' x ','h',html.Sub(current_node_id),html.Sup(f'({iter_num})')])
    text_1=flatten_lst([['W',html.Sup(f'({iter_num+1})'),'h',html.Sub(f'({i})'),html.Sup(f'({iter_num})'),f'/sqrt(deg({current_node_id})deg({i}))','+'] for i in [current_node_id]+neighbors])[:-1]
    
    # text_2=flatten_lst([f'{weight_1}×','(',
    #         flatten_lst([['{:.3f}'.format(gcn_G_fixed.nodes[i]['feature']),'+'] for i in neighbors])[:-1],')','/',str(len(neighbors)),
    #         '+',f'{weight_2}',' x ','{:.3f}'.format(gcn_G_fixed.nodes[current_node_id]['feature'])
    #         ])
    current_node_degree=gcn_G_fixed.degree[current_node_id]+1
    text_2=flatten_lst([[f'{weight_1}×','{:.3f}'.format(gcn_G_fixed.nodes[i]['feature']),f'/sqrt({current_node_degree}x{gcn_G_fixed.degree[i]+1})','+'] for i in [current_node_id]+neighbors])[:-1]
    # result=round(weight_1*sum([gcn_G_fixed.nodes[i]['feature'] for i in neighbors])/len(neighbors)+weight_2*gcn_G_fixed.nodes[current_node_id]['feature'],3)
    result=round(weight_1*sum([gcn_G_fixed.nodes[i]['feature']/np.sqrt(current_node_degree*(gcn_G_fixed.degree[i]+1)) for i in [current_node_id]+neighbors]),3)
    # print(result)
    
    content=html.Div([
        html.Span(['h',html.Sup(f'({iter_num+1})'),html.Sub(data['id']),'=','f(']+text_1+[')']),
        html.Br(),
        html.Span(['=f(']+text_2+[')']),
        html.Br(),
        html.Span(['=f(']+[result]+[')']),
        html.Br(),
        html.Span([f'ReLU({result})=']+[max(result,0)]),
        ],
        style={'font-size':'24px'})    

    return content    

@callback(
    Output('gcn_formula_update', 'children'),
    Output('gcn_current_node_id', 'children'),
    Output('gcn_weight_1_iter', 'children'),
    # Output('gcn_weight_2_iter', 'children'),
    Output('gcn_button-update', "n_clicks"),
    Output('gcn_button-undo', "n_clicks"),    
    Input('gcn_graph_fixed', 'tapNodeData'),
    Input('gcn_weight_1','value'),
    # Input('gcn_weight_2','value'),
    Input('gcn_button-update', "n_clicks"),
    Input('gcn_button-reset','n_clicks'),
    Input('gcn_button-undo', "n_clicks"),    
    )
def gcn_formula_update(data,weight_1,click_update,_,click_undo):    
    global gcn_G_fixed    
    global gcn_original_G
            
    if ctx.triggered_id == "gcn_button-reset":  
        # gcn_G_fixed=GraphConstructor.data_networkx()
        gcn_G_fixed=gcn_original_G
        data=gcn_G_fixed.nodes['A']     
        
        current_node_id=data['id']
        iter_num=1
        iter_info,weight_1_iter_info =gcn_iter_info_update(current_node_id,iter_num)
        return no_update,iter_info,weight_1_iter_info,0,0
    
    iter_num=abs(click_update-click_undo)+1
            
    if data is None:        
        data=gcn_G_fixed.nodes['A']           

    if data:
        content=gcn_formula_update_content(gcn_G_fixed,data,weight_1,iter_num-1)        
        current_node_id=data['id']
        iter_info,weight_1_iter_info=gcn_iter_info_update(current_node_id,iter_num)
        

        return content,iter_info,weight_1_iter_info,no_update,no_update      

@callback(
    Output('gcn_graph_fixed', 'elements'),
    Output('gcn_button-undo', 'disabled'),
    Input('gcn_button-update', "n_clicks"),
    Input('gcn_button-reset','n_clicks'),
    Input('gcn_button-undo', "n_clicks"),         
    Input('gcn_button-randomize', "n_clicks"),  
    Input('gcn_weight_1','value'),
    # Input('gcn_weight_2','value'),
    )
def gcn_update_all_nodes(click_update,_,click_undo,click_randomize,weight_1): 
    global gcn_update_stack
    global gcn_G_fixed
    global gcn_original_G
    global gcn_data_GC   

    if ctx.triggered_id == "gcn_button-randomize":  
        rnd_G=Random_Graph()
        gcn_G_fixed=rnd_G.random_G
        gcn_data_GC=rnd_G.random_G_cytoscape['elements']     
        gcn_original_G=copy.deepcopy(gcn_G_fixed)
        return gcn_data_GC,True      
    
    # # print('---',click_update,click_undo)
    if ctx.triggered_id == "gcn_button-reset":      
        # gcn_G_fixed=gcn_original_G
        return gcn_data_GC,True   
    
    if click_update is None:
        raise PreventUpdate
    elif click_update==0:
        return gcn_data_GC,True      
    
    if ctx.triggered_id == "gcn_button-update":      
        G_pyg_=from_networkx(gcn_G_fixed)
        data=Data(x=G_pyg_.feature.reshape(-1,1).to(torch.float), edge_index=G_pyg_.edge_index)
        
        net=GCNConv(1,1,normalize=True,bias=False)
        list(net.parameters())[0].data.fill_(weight_1)
        y=net(data.x,data.edge_index).detach().numpy().reshape(-1) 
 
        attrs_feature={k:{'feature':v} for k,v in zip(list(gcn_G_fixed.nodes()),y)}        
        attrs_label={k: {'label':f'{k}({v:.3f})'} for k,v in zip(list(gcn_G_fixed.nodes()),y)}
        
        nx.set_node_attributes(gcn_G_fixed, attrs_feature)
        nx.set_node_attributes(gcn_G_fixed, attrs_label)
        
        gcn_G_fixed_cytoscape=nx.cytoscape_data(gcn_G_fixed) 
        gcn_update_stack['elements'].append(gcn_G_fixed_cytoscape['elements'])       
        gcn_update_stack['attrs'].append([attrs_feature,attrs_label])

        return gcn_G_fixed_cytoscape['elements'],False   
        
    if ctx.triggered_id == "gcn_button-undo": 
        if len(gcn_update_stack['elements'])>1:
            attrs_feature,attrs_label=gcn_update_stack['attrs'].pop()
            nx.set_node_attributes(gcn_G_fixed, attrs_feature)
            nx.set_node_attributes(gcn_G_fixed, attrs_label)
            return gcn_update_stack['elements'].pop(),False    
        if click_update-click_undo<2:            
            gcn_G_fixed=GraphConstructor.data_networkx()   
            return gcn_data_GC,True
        else:                        
            gcn_G_fixed=GraphConstructor.data_networkx() 
            return gcn_data_GC,True        
        
#%%tab_GAT
gat_G_fixed=GraphConstructor.data_networkx()
gat_update_stack={'elements':[],'attrs':[]}
gat_data_GC=GraphConstructor.data()
gat_original_G=GraphConstructor.data_networkx()  

def tab_gat():      
    tab_div=dbc.Container([
        html.Br(),
        html.Div([
            dbc.Button('Reset', id='gat_button-reset',color="dark",n_clicks=0,className="me-1"),
            dbc.Button('Undo Last Update', id='gat_button-undo',color="dark",n_clicks=0,className="me-1",disabled=True),
            dbc.Button('Update All Nodes', id='gat_button-update',color="dark",n_clicks=0,className="me-1"),
            dbc.Button('Randomize Graph', id='gat_button-randomize',color="dark",n_clicks=0,className="me-1"),
            ],
            # align="center",            
            ),
        dbc.Row([
            dbc.Col([
                html.P(id='gat_weight_1_iter'),       
                dcc.Slider(-10, 10, 0.1, value=1, marks=None,tooltip={"placement": "bottom", "always_visible": True},id='gat_weight_1'),
                html.P(id='gat_weight_2_iter'), 
                dcc.Slider(-1, 1, 0.01, value=1, marks=None,tooltip={"placement": "bottom", "always_visible": True},id='gat_weight_2'),                
                ],),       
            dbc.Col([
                html.B('Initial Graph'),     
                cyto.Cytoscape(
                        id='gat_graph_fixed',
                        elements=GraphConstructor.data(),
                        style={'width': '100%', 'height': '500px'},
                        layout={
                            'name': 'cose'
                        },
                        stylesheet=GraphConstructor.stylesheet(),
                    ),       
                ],),        
            ], 
            align="center",
            ),     
        dbc.Row([
            dbc.Col([
            html.P(id='gat_current_node_id'),
            html.Div(id='gat_formula_update'),
            html.P(dcc.Markdown('''
                                ---
                                
                                We have omitted the superscripts on the attention weights for clarity.
                                
                                Here, $f$ is LeakyReLU: $LeakyReLU (x)=\max (0, x)+ negative\_slope * \min (0, x)$,  
                                
                                Note that the weight $W^{(k)}$ and $A_W^{(k)}$ are shared across all nodes!         
                                ''',mathjax=True)),
                ]),    
            # dbc.Col([
            #     html.P([
            #         ]),
            #     ]),            
            ], 
            align="center",
            ),    
           dbc.Row([          
               html.Div(id='gat_cytoscape-tapNodeData-output'),
               ]),
    ],
    fluid=True,
    )    
    return tab_div     
     
def gat_iter_info_update(current_node_id,iter_num):
        iter_info=[
            html.B(f'Next Update (Iteration {iter_num}):'),
            html.Br(),
            html.Span(f'Equation for Node {current_node_id}:')
            ]
        
        weight_1_iter_info=[
            html.B('Parameters for Next Update'),
            html.Br(),
            html.Span(['W',html.Sup(f'({iter_num})')])
            ] 
        
        weight_2_iter_info=[
            html.Span(['A',html.Sub('W'),html.Sup(f'({iter_num})')])
            ]
        
        return iter_info,weight_1_iter_info,weight_2_iter_info 
    
def gat_formula_update_content(gcn_G_fixed,data,weight_1,weight_2,iter_num):
    current_node_id=data['id']
    neighbors=list(gcn_G_fixed.neighbors(current_node_id))
    current_node_degree=gcn_G_fixed.degree[current_node_id]+1

    e_node=[weight_2*weight_1*(gcn_G_fixed.nodes[i]['feature']+gcn_G_fixed.nodes[current_node_id]['feature']) for i in [current_node_id]+neighbors]
    LReLU = nn.LeakyReLU(negative_slope=0.2) #0.2
    e_node_LeakyReLU=[LReLU(torch.tensor(e).to(torch.float)) for e in e_node]                     
                                
    text_1=flatten_lst([['e',html.Sub(f'{current_node_id}'),html.Sub(f'{i}'),'=LeakyReLU(',
                         'A',html.Sub('W'),html.Sup(f'({iter_num+1})'),'(','W',html.Sup(f'({iter_num+1})'),
                         'h',html.Sub(f'{current_node_id}'),html.Sup(f'({iter_num})'),' + ','W',html.Sup(f'({iter_num+1})'),'h',html.Sub(f'{i}'),html.Sup(f'({iter_num})'),
                         '))=LeakyReLU(',f'{weight_2}','(',f'{weight_1}',' x ', '{:.3f}'.format(gcn_G_fixed.nodes[current_node_id]['feature']),
                         ' + ', f'{weight_1}',' x ', '{:.3f}'.format(gcn_G_fixed.nodes[i]['feature']),'))','=LeakyReLU(',  
                         '{:.3f}'.format(e),')=','{:.3f}'.format(LReLU(torch.tensor(e).to(torch.float))),
                         html.Br(),] for i,e in zip([current_node_id]+neighbors,e_node)])
    
    softmax=nn.Softmax(dim=0)
    a_node=softmax(torch.tensor(e_node_LeakyReLU))
    
    text_2=flatten_lst([['a',html.Sub(f'{current_node_id}'),html.Sub(f'{i}'),'=exp(e',
        html.Sub(f'{current_node_id}'),html.Sub(f'{i}'),')/(',flatten_lst([['exp(e',html.Sub(f'{current_node_id}'),html.Sub(f'{i}'),')',' + '] for i in [current_node_id]+neighbors])[:-1],')',
        '=','{:.3f}'.format(a),
        html.Br(),] for i,a in zip([current_node_id]+neighbors,a_node)])
    
    result=sum([a*gcn_G_fixed.nodes[i]['feature'] for i,a in zip([current_node_id]+neighbors,a_node)])*weight_1
    
    text_3=flatten_lst([flatten_lst(['h',html.Sub(f'{current_node_id}'),html.Sup(f'({iter_num+1})'),'=','f(W',html.Sup(f'({iter_num+1})'), ' x (',flatten_lst([['a',html.Sub(f'{current_node_id}'),html.Sub(f'{i}'), ' x ',
                       'h',html.Sub(f'{i}'),html.Sup(f'({iter_num})'),' + '] for i in [current_node_id]+neighbors])[:-1],'))']),html.Br(),
                        ['=f(','{:.3f}'.format(weight_1),' x (',flatten_lst([['{:.3f}'.format(a), ' x ','{:.3f}'.format(gcn_G_fixed.nodes[i]['feature']), ' + '] for i,a in zip([current_node_id]+neighbors,a_node)])[:-1]],html.Br(),
                        ['=f(','{:.3f}'.format(result),')'],html.Br(),
                        ['=LeakyReLU(','{:.3f}'.format(result),')'],html.Br(),
                        ['=','{:.3f}'.format(LReLU(result))]
                        ])
    
    
    content=html.Div([
        html.Span(text_3),
        html.Br(),
        html.Span(['with attention weights a', html.Sub(f'{current_node_id}'),'computed as:'],style={'font-size':'16px'}),
        html.Br(),
        html.Span(text_2),
        html.Br(),
        html.Span(['where the unnormalized attention weights e', html.Sub(f'{current_node_id}'),' are given by:'],style={'font-size':'16px'}),
        html.Br(),
        html.Span(text_1)
        
        ],
        style={'font-size':'24px'})    

    return content       
        
@callback(
    Output('gat_formula_update', 'children'),
    Output('gat_current_node_id', 'children'),
    Output('gat_weight_1_iter', 'children'),
    Output('gat_weight_2_iter', 'children'),
    Output('gat_button-update', "n_clicks"),
    Output('gat_button-undo', "n_clicks"),    
    Input('gat_graph_fixed', 'tapNodeData'),
    Input('gat_weight_1','value'),
    Input('gat_weight_2','value'),
    Input('gat_button-update', "n_clicks"),
    Input('gat_button-reset','n_clicks'),
    Input('gat_button-undo', "n_clicks"),    
    )
def gat_formula_update(data,weight_1,weight_2,click_update,_,click_undo):    
    global gat_G_fixed    
    global gat_original_G
            
    if ctx.triggered_id == "gat_button-reset":  
        # gat_G_fixed=GraphConstructor.data_networkx()
        gat_G_fixed=gat_original_G
        data=gat_G_fixed.nodes['A']     
        
        current_node_id=data['id']
        iter_num=1
        iter_info,weight_1_iter_info,weight_2_iter_info=gat_iter_info_update(current_node_id,iter_num)
        return no_update,iter_info,weight_1_iter_info,weight_2_iter_info ,0,0
    
    iter_num=abs(click_update-click_undo)+1
            
    if data is None:        
        data=gat_G_fixed.nodes['A']           

    if data:
        content=gat_formula_update_content(gat_G_fixed,data,weight_1,weight_2,iter_num-1)        
        current_node_id=data['id']
        iter_info,weight_1_iter_info,weight_2_iter_info=gat_iter_info_update(current_node_id,iter_num)
        

        return content,iter_info,weight_1_iter_info,weight_2_iter_info,no_update,no_update  

@callback(
    Output('gat_graph_fixed', 'elements'),
    Output('gat_button-undo', 'disabled'),
    Input('gat_button-update', "n_clicks"),
    Input('gat_button-reset','n_clicks'),
    Input('gat_button-undo', "n_clicks"),         
    Input('gat_button-randomize', "n_clicks"),  
    Input('gat_weight_1','value'),
    Input('gat_weight_2','value'),
    )
def gat_update_all_nodes(click_update,_,click_undo,click_randomize,weight_1,weight_2): 
    global gat_update_stack
    global gat_G_fixed
    global gat_original_G
    global gat_data_GC   

    if ctx.triggered_id == "gat_button-randomize":  
        rnd_G=Random_Graph()
        gat_G_fixed=rnd_G.random_G
        gat_data_GC=rnd_G.random_G_cytoscape['elements']     
        gat_original_G=copy.deepcopy(gat_G_fixed)
        return gat_data_GC,True      
    
    # # print('---',click_update,click_undo)
    if ctx.triggered_id == "gat_button-reset":      
        # gat_G_fixed=gat_original_G
        return gat_data_GC,True   
    
    if click_update is None:
        raise PreventUpdate
    elif click_update==0:
        return gat_data_GC,True      
    
    if ctx.triggered_id == "gat_button-update":      
        G_pyg_=from_networkx(gat_G_fixed)
        data=Data(x=G_pyg_.feature.reshape(-1,1).to(torch.float), edge_index=G_pyg_.edge_index)
        
        net=GATConv(data.num_features,1,normalize=True,bias=False,negative_slope=0.2)
        for para in list(net.parameters()):   
            para.data.fill_(1.0)
        y=net(data.x,data.edge_index).detach().numpy().reshape(-1) 
 
        attrs_feature={k:{'feature':v} for k,v in zip(list(gat_G_fixed.nodes()),y)}        
        attrs_label={k: {'label':f'{k}({v:.3f})'} for k,v in zip(list(gat_G_fixed.nodes()),y)}
        
        nx.set_node_attributes(gat_G_fixed, attrs_feature)
        nx.set_node_attributes(gat_G_fixed, attrs_label)
        
        gat_G_fixed_cytoscape=nx.cytoscape_data(gat_G_fixed) 
        gat_update_stack['elements'].append(gat_G_fixed_cytoscape['elements'])       
        gat_update_stack['attrs'].append([attrs_feature,attrs_label])

        return gat_G_fixed_cytoscape['elements'],False   
        
    if ctx.triggered_id == "gat_button-undo": 
        if len(gat_update_stack['elements'])>1:
            attrs_feature,attrs_label=gat_update_stack['attrs'].pop()
            nx.set_node_attributes(gat_G_fixed, attrs_feature)
            nx.set_node_attributes(gat_G_fixed, attrs_label)
            return gat_update_stack['elements'].pop(),False    
        if click_update-click_undo<2:            
            gat_G_fixed=GraphConstructor.data_networkx()   
            return gat_data_GC,True
        else:                        
            gat_G_fixed=GraphConstructor.data_networkx() 
            return gat_data_GC,True              

#%% tab_GraphSAGE
sage_G_fixed=GraphConstructor.data_networkx()
sage_update_stack={'elements':[],'attrs':[]}
sage_data_GC=GraphConstructor.data()
sage_original_G=GraphConstructor.data_networkx()  


def tab_sage():      
    tab_div=dbc.Container([
        html.Br(),
        html.Div([
            dbc.Button('Reset', id='sage_button-reset',color="dark",n_clicks=0,className="me-1"),
            dbc.Button('Undo Last Update', id='sage_button-undo',color="dark",n_clicks=0,className="me-1",disabled=True),
            dbc.Button('Update All Nodes', id='sage_button-update',color="dark",n_clicks=0,className="me-1"),
            dbc.Button('Randomize Graph', id='sage_button-randomize',color="dark",n_clicks=0,className="me-1"),
            ],
            # align="center",            
            ),
        dbc.Row([
            dbc.Col([
                html.P(id='sage_weight_1_iter'),       
                dcc.Slider(-10, 10, 0.1, value=1, marks=None,tooltip={"placement": "bottom", "always_visible": True},id='sage_weight_1'),
                html.P(id='sage_weight_2_iter'), 
                dcc.Slider(-10, 10, 0.1, value=1, marks=None,tooltip={"placement": "bottom", "always_visible": True},id='sage_weight_2'),                
                ],),       
            dbc.Col([
                html.B('Initial Graph'),     
                cyto.Cytoscape(
                        id='sage_graph_fixed',
                        elements=GraphConstructor.data(),
                        style={'width': '100%', 'height': '500px'},
                        layout={
                            'name': 'cose'
                        },
                        stylesheet=GraphConstructor.stylesheet(),
                    ),       
                ],),        
            ], 
            align="center",
            ),     
        dbc.Row([
            dbc.Col([
            html.P(id='sage_current_node_id'),
            html.Div(id='sage_formula_update'),
            html.P(dcc.Markdown('''
                                ---
                                
                                The calculation of LSTM directly calls the LSTM method of the pytorch library.
                                
                                Note that the weight $W^{(k)}$ and $B^{(k)}$ are shared across all nodes!         
                                ''',mathjax=True)),
                ]),    
            # dbc.Col([
            #     html.P([
            #         ]),
            #     ]),            
            ], 
            align="center",
            ),    
           dbc.Row([          
               html.Div(id='sage_cytoscape-tapNodeData-output'),
               ]),
    ],
    fluid=True,
    )    
    return tab_div  

def sage_iter_info_update(current_node_id,iter_num):
        iter_info=[
            html.B(f'Next Update (Iteration {iter_num}):'),
            html.Br(),
            html.Span(f'Equation for Node {current_node_id}:')
            ]
        
        weight_1_iter_info=[
            html.B('Parameters for Next Update'),
            html.Br(),
            html.Span(['W',html.Sup(f'({iter_num})')])
            ] 
        
        weight_2_iter_info=[
            html.Span(['B',html.Sup(f'({iter_num})')])
            ]
        
        return iter_info,weight_1_iter_info,weight_2_iter_info 
    
def LSTM4G(G):
    G_pyg=from_networkx(G)
    G_pyg= Data(x=G_pyg.feature.reshape(-1,1).to(torch.float), edge_index=G_pyg.edge_index)
    G_pyg=G_pyg.sort(sort_by_row=False)
    x=torch.tensor([G_pyg.x[i] for i in G_pyg.edge_index[0]]).reshape(-1,1)
    x,_=to_dense_batch(x,G_pyg.edge_index[1])
    lstm=LSTM(1, 1, batch_first=True)
    for para in list(lstm.parameters()):   
        para.data.fill_(1.0)

    lstm_out=lstm(x)[0][:,-1]    
    return lstm_out,x
    
    
def sage_formula_update_content(gcn_G_fixed,data,weight_1,weight_2,iter_num):
    current_node_id=data['id']
        
    lstm_out,x=LSTM4G(gcn_G_fixed)
    
    id2int=lambda id_text: ord(id_text)-65
    current_lstmout=lstm_out[id2int(current_node_id)][0].detach().item()
    text_1=['h',html.Sup(f'({iter_num+1})'),'=LSTM(','{}'.format(x[id2int(current_node_id)].flatten().tolist()),')=','{:.5f}'.format(current_lstmout)]
    
    neighbors=list(gcn_G_fixed.neighbors(current_node_id))
    text_2=flatten_lst(['h',html.Sub(f'{current_node_id}'),html.Sup(f'({iter_num+1})'),'=','f(W',html.Sup(f'({iter_num+1})'), ' x LSTM[',   
                        flatten_lst([['h',html.Sub(f'{i}'),html.Sup(f'({iter_num})'),','] for i in neighbors])[:-1], '] + B', html.Sup(f'({iter_num+1})'), ' x ',   
                        'h',html.Sub(f'{current_node_id}'),html.Sup(f'({iter_num})'),                            
                        ])
    text_3=['=f(','{:.3f}'.format(weight_1),' x ','{:.5f}'.format(current_lstmout),' + ',
            '{:.3f}'.format(weight_2),' x ','{:.3f}'.format(gcn_G_fixed.nodes[current_node_id]['feature']),')']
    result=weight_1*current_lstmout+weight_2*gcn_G_fixed.nodes[current_node_id]['feature']
    text_4=['=f(','{:.3f}'.format(result),')',html.Br(),'ReLU(','{:.3f}'.format(result),')',html.Br(),'={:.3f}'.format(max(result,0))]
    
    content=html.Div([
        html.Span(text_2),
        html.Br(),
        html.Span(text_3),   
        html.Br(),
        html.Span(text_4),  
        html.Br(),
        html.Span(['Considering the sequence of neighbours as',f' {neighbors}, ', 'the LSTM computes (set all parameters to 1):'],style={'font-size':'16px'}),
        html.Br(),
        html.Span(text_1),   
        ],
        style={'font-size':'24px'})    
 
    return content           

@callback(
    Output('sage_formula_update', 'children'),
    Output('sage_current_node_id', 'children'),
    Output('sage_weight_1_iter', 'children'),
    Output('sage_weight_2_iter', 'children'),
    Output('sage_button-update', "n_clicks"),
    Output('sage_button-undo', "n_clicks"),    
    Input('sage_graph_fixed', 'tapNodeData'),
    Input('sage_weight_1','value'),
    Input('sage_weight_2','value'),
    Input('sage_button-update', "n_clicks"),
    Input('sage_button-reset','n_clicks'),
    Input('sage_button-undo', "n_clicks"),    
    )
def sage_formula_update(data,weight_1,weight_2,click_update,_,click_undo):    
    global sage_G_fixed    
    global sage_original_G
            
    if ctx.triggered_id == "sage_button-reset":  
        # sage_G_fixed=GraphConstructor.data_networkx()
        sage_G_fixed=sage_original_G
        data=sage_G_fixed.nodes['A']     
        
        current_node_id=data['id']
        iter_num=1
        iter_info,weight_1_iter_info,weight_2_iter_info=sage_iter_info_update(current_node_id,iter_num)
        return no_update,iter_info,weight_1_iter_info,weight_2_iter_info ,0,0
    
    iter_num=abs(click_update-click_undo)+1
            
    if data is None:        
        data=sage_G_fixed.nodes['A']           

    if data:
        content=sage_formula_update_content(sage_G_fixed,data,weight_1,weight_2,iter_num-1)        
        current_node_id=data['id']
        iter_info,weight_1_iter_info,weight_2_iter_info=sage_iter_info_update(current_node_id,iter_num)
        

        return content,iter_info,weight_1_iter_info,weight_2_iter_info,no_update,no_update  

@callback(
    Output('sage_graph_fixed', 'elements'),
    Output('sage_button-undo', 'disabled'),
    Input('sage_button-update', "n_clicks"),
    Input('sage_button-reset','n_clicks'),
    Input('sage_button-undo', "n_clicks"),         
    Input('sage_button-randomize', "n_clicks"),  
    Input('sage_weight_1','value'),
    Input('sage_weight_2','value'),
    )
def sage_update_all_nodes(click_update,_,click_undo,click_randomize,weight_1,weight_2): 
    global sage_update_stack
    global sage_G_fixed
    global sage_original_G
    global sage_data_GC   

    if ctx.triggered_id == "sage_button-randomize":  
        rnd_G=Random_Graph()
        sage_G_fixed=rnd_G.random_G
        sage_data_GC=rnd_G.random_G_cytoscape['elements']     
        sage_original_G=copy.deepcopy(sage_G_fixed)
        return sage_data_GC,True      
    
    # # print('---',click_update,click_undo)
    if ctx.triggered_id == "sage_button-reset":      
        # sage_G_fixed=sage_original_G
        return sage_data_GC,True   
    
    if click_update is None:
        raise PreventUpdate
    elif click_update==0:
        return sage_data_GC,True      
    
    if ctx.triggered_id == "sage_button-update":      
        lstm_out,x=LSTM4G(sage_G_fixed)
        features=list(nx.get_node_attributes(sage_G_fixed,'feature').values())
        y=[max(weight_1*i.detach().item()+weight_2*j,0) for i,j in zip(lstm_out,features)]

        attrs_feature={k:{'feature':v} for k,v in zip(list(sage_G_fixed.nodes()),y)}        
        attrs_label={k: {'label':f'{k}({v:.3f})'} for k,v in zip(list(sage_G_fixed.nodes()),y)}
        
        nx.set_node_attributes(sage_G_fixed, attrs_feature)
        nx.set_node_attributes(sage_G_fixed, attrs_label)
        
        sage_G_fixed_cytoscape=nx.cytoscape_data(sage_G_fixed) 
        sage_update_stack['elements'].append(sage_G_fixed_cytoscape['elements'])       
        sage_update_stack['attrs'].append([attrs_feature,attrs_label])

        return sage_G_fixed_cytoscape['elements'],False   
        
    if ctx.triggered_id == "sage_button-undo": 
        if len(sage_update_stack['elements'])>1:
            attrs_feature,attrs_label=sage_update_stack['attrs'].pop()
            nx.set_node_attributes(sage_G_fixed, attrs_feature)
            nx.set_node_attributes(sage_G_fixed, attrs_label)
            return sage_update_stack['elements'].pop(),False    
        if click_update-click_undo<2:            
            sage_G_fixed=GraphConstructor.data_networkx()   
            return sage_data_GC,True
        else:                        
            sage_G_fixed=GraphConstructor.data_networkx() 
            return sage_data_GC,True          
#%% tab_GIN
gin_G_fixed=GraphConstructor.data_networkx()
gin_update_stack={'elements':[],'attrs':[]}
gin_data_GC=GraphConstructor.data()
gin_original_G=GraphConstructor.data_networkx()  

def tab_gin():      
    tab_div=dbc.Container([
        html.Br(),
        html.Div([
            dbc.Button('Reset', id='gin_button-reset',color="dark",n_clicks=0,className="me-1"),
            dbc.Button('Undo Last Update', id='gin_button-undo',color="dark",n_clicks=0,className="me-1",disabled=True),
            dbc.Button('Update All Nodes', id='gin_button-update',color="dark",n_clicks=0,className="me-1"),
            dbc.Button('Randomize Graph', id='gin_button-randomize',color="dark",n_clicks=0,className="me-1"),
            ],
            # align="center",            
            ),
        dbc.Row([
            dbc.Col([
                html.P(id='gin_weight_1_iter'),       
                dcc.Slider(-10, 10, 0.1, value=0, marks=None,tooltip={"placement": "bottom", "always_visible": True},id='gin_weight_1'),
                # html.P(id='gin_weight_2_iter'), 
                # dcc.Slider(-1, 1, 0.01, value=1, marks=None,tooltip={"placement": "bottom", "always_visible": True},id='gin_weight_2'),                
                ],),       
            dbc.Col([
                html.B('Initial Graph'),     
                cyto.Cytoscape(
                        id='gin_graph_fixed',
                        elements=GraphConstructor.data(),
                        style={'width': '100%', 'height': '500px'},
                        layout={
                            'name': 'cose'
                        },
                        stylesheet=GraphConstructor.stylesheet(),
                    ),       
                ],),        
            ], 
            align="center",
            ),     
        dbc.Row([
            dbc.Col([
            html.P(id='gin_current_node_id'),
            html.Div(id='gin_formula_update'),
            html.P(dcc.Markdown('''
                                ---
                                
                                Here, $f$ is just ReLu: $f(x)=max(x,0)$,  
                                
                                Note that the weight $ε^{(k)}$ is shared across all nodes!         
                                ''',mathjax=True)),
                ]),    
            # dbc.Col([
            #     html.P([
            #         ]),
            #     ]),            
            ], 
            align="center",
            ),    
           dbc.Row([          
               html.Div(id='gin_cytoscape-tapNodeData-output'),
               ]),
    ],
    fluid=True,
    )    
    return tab_div     

def gin_iter_info_update(current_node_id,iter_num):
        iter_info=[
            html.B(f'Next Update (Iteration {iter_num}):'),
            html.Br(),
            html.Span(f'Equation for Node {current_node_id}:')
            ]
        
        weight_1_iter_info=[
            html.B('Parameters for Next Update'),
            html.Br(),
            html.Span(['ε',html.Sup(f'({iter_num})')])  # dcc.Markdown('''$\epsilon$''',mathjax=True)
            ] 
        
        # weight_2_iter_info=[
        #     html.Span(['B',html.Sup(f'({iter_num})')])
        #     ]
        
        return iter_info,weight_1_iter_info 
    
def gin_formula_update_content(G_fixed,data,weight_1,iter_num):
    current_node_id=data['id']
    neighbors=list(G_fixed.neighbors(data['id']))
    text=flatten_lst([['h',html.Sub(f'({i})'),html.Sup(f'({iter_num})'),'+'] for i in neighbors])[:-1]
    
    
    text_1=flatten_lst([['{:.3f}'.format(G_fixed.nodes[i]['feature']),'+'] for i in neighbors])[:-1]
    result=sum([G_fixed.nodes[i]['feature'] for i in neighbors])+(weight_1+1)*G_fixed.nodes[current_node_id]['feature']
    
    content=html.Div([
        html.Span(['h',html.Sup(f'({iter_num+1})'),html.Sub(data['id']),'=','f((']+text+[') + (1 + ε',html.Sup(f'({iter_num+1})'),') x ',
                  'h',html.Sub(f'{current_node_id}'),html.Sup(f'({iter_num})'),]),
        html.Br(),
        html.Span(['=f((']+text_1+[') + (1 + ']+[f'{weight_1}',') x ',G_fixed.nodes[current_node_id]['feature'],')']),
        # html.Span(text_1),
        html.Br(),
        html.Span(['=f(']+[f'{result:.3f}']+[')']),
        html.Br(),
        html.Span([f'ReLU({result})=']+[max(result,0)]),
        ],
        style={'font-size':'24px'})    
    
    return content    

@callback(
    Output('gin_formula_update', 'children'),
    Output('gin_current_node_id', 'children'),
    Output('gin_weight_1_iter', 'children'),
    # Output('gin_weight_2_iter', 'children'),
    Output('gin_button-update', "n_clicks"),
    Output('gin_button-undo', "n_clicks"),    
    Input('gin_graph_fixed', 'tapNodeData'),
    Input('gin_weight_1','value'),
    # Input('gin_weight_2','value'),
    Input('gin_button-update', "n_clicks"),
    Input('gin_button-reset','n_clicks'),
    Input('gin_button-undo', "n_clicks"),    
    )
def gin_formula_update(data,weight_1,click_update,_,click_undo):    
    global gin_G_fixed    
    global gin_original_G
            
    if ctx.triggered_id == "gin_button-reset":  
        # gin_G_fixed=GraphConstructor.data_networkx()
        gin_G_fixed=gin_original_G
        data=gin_G_fixed.nodes['A']     
        
        current_node_id=data['id']
        iter_num=1
        iter_info,weight_1_iter_info =gin_iter_info_update(current_node_id,iter_num)
        return no_update,iter_info,weight_1_iter_info,0,0
    
    iter_num=abs(click_update-click_undo)+1
            
    if data is None:        
        data=gin_G_fixed.nodes['A']           

    if data:
        content=gin_formula_update_content(gin_G_fixed,data,weight_1,iter_num-1)        
        current_node_id=data['id']
        iter_info,weight_1_iter_info=gin_iter_info_update(current_node_id,iter_num)
        

        return content,iter_info,weight_1_iter_info,no_update,no_update   

@callback(
    Output('gin_graph_fixed', 'elements'),
    Output('gin_button-undo', 'disabled'),
    Input('gin_button-update', "n_clicks"),
    Input('gin_button-reset','n_clicks'),
    Input('gin_button-undo', "n_clicks"),         
    Input('gin_button-randomize', "n_clicks"),  
    Input('gin_weight_1','value'),
    # Input('gin_weight_2','value'),
    )
def gin_update_all_nodes(click_update,_,click_undo,click_randomize,weight_1): 
    global gin_update_stack
    global gin_G_fixed
    global gin_original_G
    global gin_data_GC   

    if ctx.triggered_id == "gin_button-randomize":  
        rnd_G=Random_Graph()
        gin_G_fixed=rnd_G.random_G
        gin_data_GC=rnd_G.random_G_cytoscape['elements']     
        gin_original_G=copy.deepcopy(gin_G_fixed)
        return gin_data_GC,True      
    
    # # print('---',click_update,click_undo)
    if ctx.triggered_id == "gin_button-reset":      
        # gin_G_fixed=gin_original_G
        return gin_data_GC,True   
    
    if click_update is None:
        raise PreventUpdate
    elif click_update==0:
        return gin_data_GC,True      
    
    if ctx.triggered_id == "gin_button-update":      
        G_pyg_=from_networkx(gin_G_fixed)
        data=Data(x=G_pyg_.feature.reshape(-1,1).to(torch.float), edge_index=G_pyg_.edge_index)
        
        nn_sequential=Sequential(Linear(data.num_node_features, 1,bias=False),torch.nn.ReLU())    
        net=GINConv(nn_sequential,eps=weight_1)     
        for para in list(net.parameters()):   
            para.data.fill_(1.0)       

        y=net(data.x,data.edge_index).detach().numpy().reshape(-1) 
 
        attrs_feature={k:{'feature':v} for k,v in zip(list(gin_G_fixed.nodes()),y)}        
        attrs_label={k: {'label':f'{k}({v:.3f})'} for k,v in zip(list(gin_G_fixed.nodes()),y)}
        
        nx.set_node_attributes(gin_G_fixed, attrs_feature)
        nx.set_node_attributes(gin_G_fixed, attrs_label)
        
        gin_G_fixed_cytoscape=nx.cytoscape_data(gin_G_fixed) 
        gin_update_stack['elements'].append(gin_G_fixed_cytoscape['elements'])       
        gin_update_stack['attrs'].append([attrs_feature,attrs_label])

        return gin_G_fixed_cytoscape['elements'],False   
        
    if ctx.triggered_id == "gin_button-undo": 
        if len(gin_update_stack['elements'])>1:
            attrs_feature,attrs_label=gin_update_stack['attrs'].pop()
            nx.set_node_attributes(gin_G_fixed, attrs_feature)
            nx.set_node_attributes(gin_G_fixed, attrs_label)
            return gin_update_stack['elements'].pop(),False    
        if click_update-click_undo<2:            
            gin_G_fixed=GraphConstructor.data_networkx()   
            return gin_data_GC,True
        else:                        
            gin_G_fixed=GraphConstructor.data_networkx() 
            return gin_data_GC,True        
#%%


if __name__ == '__main__':
    app.run(debug=True)