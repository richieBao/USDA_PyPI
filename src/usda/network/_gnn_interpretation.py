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
from torch.nn import Linear
import numpy as np
import random
import copy

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
  
G_fixed=GraphConstructor.data_networkx()
update_stack={'elements':[],'attrs':[]}
data_GC=GraphConstructor.data()
original_G=GraphConstructor.data_networkx()   

#%%
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
                                
                                Note that the weight $W^{(1)}$ is shared across all nodes!         
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

#%%
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
   
#%%      
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
            G_fixed=original_G
            return data_GC,True
        else:            
            G_fixed=original_G
            return data_GC,True        
    
#%%
def tab_gcn():      
    tab_div=dbc.Container([
        html.Br(),
        html.Div([
            dbc.Button('Reset', id='button-reset',color="dark",n_clicks=0,className="me-1"),
            dbc.Button('Undo Last Update', id='button-undo',color="dark",n_clicks=0,className="me-1",disabled=True),
            dbc.Button('Update All Nodes', id='button-update',color="dark",n_clicks=0,className="me-1"),
            dbc.Button('Randomize Graph', id='button-randomize',color="dark",n_clicks=0,className="me-1"),
            ],
            # align="center",            
            ),
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
                                
                                Note that the weight $W^{(1)}$ is shared across all nodes!         
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




#%%
if __name__ == '__main__':
    app.run(debug=True)