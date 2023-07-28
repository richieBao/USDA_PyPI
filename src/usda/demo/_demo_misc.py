# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 20:49:45 2023

@author: richie bao
migrated from: Isotonic Regression https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_isotonic_regression.html#sphx-glr-auto-examples-miscellaneous-plot-isotonic-regression-py
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_random_state

from scipy.linalg import eigh,eig
from sklearn.datasets import make_moons,make_circles

import time as time
from sklearn.datasets import make_swiss_roll
import mpl_toolkits.mplot3d 
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances
from sklearn import manifold
import matplotlib as mpl
from matplotlib import pyplot as plt, colors
import networkx as nx

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.optimize import fsolve

def demo_isotonic_regression(n=100,figsize=(12, 6),markersize=12):
    x = np.arange(n)
    rs = check_random_state(0)
    y = rs.randint(-50, 50, size=(n,)) + 50.0 * np.log1p(np.arange(n))    
    
    ir = IsotonicRegression(out_of_bounds="clip")
    y_ = ir.fit_transform(x, y)
    
    lr = LinearRegression()
    lr.fit(x[:, np.newaxis], y)  # x needs to be 2d for LinearRegression    

    segments = [[[i, y[i]], [i, y_[i]]] for i in range(n)]
    lc = LineCollection(segments, zorder=0)
    lc.set_array(np.ones(len(y)))
    lc.set_linewidths(np.full(n, 0.5))
    
    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=figsize)
    
    ax0.plot(x, y, "C0.", markersize=markersize)
    ax0.plot(x, y_, "C1.-", markersize=markersize)
    ax0.plot(x, lr.predict(x[:, np.newaxis]), "C2-")
    ax0.add_collection(lc)
    ax0.legend(("Training data", "Isotonic fit", "Linear fit"), loc="lower right")
    ax0.set_title("Isotonic regression fit on noisy data (n=%d)" % n)
    
    x_test = np.linspace(-10, 110, 1000)
    ax1.plot(x_test, ir.predict(x_test), "C1-")
    ax1.plot(ir.X_thresholds_, ir.y_thresholds_, "C1.", markersize=markersize)
    ax1.set_title("Prediction function (%d thresholds)" % len(ir.X_thresholds_))
    
    plt.show()
    
    return x,y,ir

def demo_eigvals_eigvecs_2d(figsize=(10,10)):

    X = np.arange(-10, 10, 1)
    Y = np.arange(-10, 10, 1)
    U, V = np.meshgrid(X, Y)

    T=np.array([[2,1],[1,2]])
    eUV=np.matmul(T,np.stack([U.ravel(),V.ravel()]))
    eigvals, eigvecs=eigh(T)
    print(f'A:\n{T}\neigenvalue:\n{eigvals}\neigenvector:\n{eigvecs}')    
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.hlines(y=0, xmin=-30, xmax=30, color='gray')
    ax.vlines(x=0, ymin=-30, ymax=30, color='gray')
    scale=15
    ax.arrow(0,0,eigvecs[0][0]*scale,eigvecs[0][1]*scale,width=0.1,head_length=0.5,color='orangered')
    ax.arrow(0,0,eigvecs[1][0]*scale,eigvecs[1][1]*scale,width=0.1,head_length=0.5,color='dodgerblue')

    ax.scatter(U,V,s=30,c='gray')
    ax.scatter(eUV[0],eUV[1],s=5,c='k')

    ax.arrow(8,8,1,1,width=0.5,head_length=0.5,color='dodgerblue',alpha=0.5)
    bbox_props = dict(boxstyle="round", fc="k", ec="0.5", alpha=0.8)
    ax.text(7, 8, "1", ha="center", va="center", size=8,bbox=bbox_props, weight='bold',color='w')

    ax.arrow(24,24,eigvals[1],eigvals[1],width=0.5,head_length=0.5,color='dodgerblue',alpha=0.5)
    ax.text(21, 24, "1 (len=3)", ha="center", va="center", size=8,bbox=bbox_props, weight='bold',color='w')

    ax.arrow(-8,8,-1,1,width=0.5,head_length=0.5,color='orangered',alpha=0.5)
    ax.text(-9, 7, "2", ha="center", va="center", size=8,bbox=bbox_props, weight='bold',color='w')

    ax.arrow(0,8,-1,1,width=0.5,head_length=0.5,color='orangered',alpha=0.5)
    ax.text(-1, 7, "3", ha="center", va="center", size=8,bbox=bbox_props, weight='bold',color='w')

    ax.arrow(10,17,-1,1,width=0.5,head_length=0.5,color='orangered',alpha=0.5)
    ax.text(9, 16, "3", ha="center", va="center", size=8,bbox=bbox_props, weight='bold',color='w')

    ax.arrow(8,4,1,1,width=0.5,head_length=0.5,color='dodgerblue',alpha=0.5)
    ax.text(8,2.7, "4", ha="center", va="center", size=8,bbox=bbox_props, weight='bold',color='w')

    ax.arrow(20,16,eigvals[1],eigvals[1],width=0.5,head_length=0.5,color='dodgerblue',alpha=0.5)
    ax.text(21, 15, "4 (len=3)", ha="center", va="center", size=8,bbox=bbox_props, weight='bold',color='w')

    ax.arrow(-9,0,-1,0,width=0.5,head_length=0.5,color='purple',alpha=0.5)
    ax.text(-9, -1, "5", ha="center", va="center", size=8,bbox=bbox_props, weight='bold',color='w')

    n_5=np.matmul(T,[-9,0])
    n_5_d=np.matmul(T,[-1,0])
    ax.arrow(n_5[0],n_5[1],n_5_d[0],n_5_d[1],width=0.5,head_length=0.5,color='purple',alpha=0.5)
    ax.text(n_5[0]+1,n_5[1], "5", ha="center", va="center", size=8,bbox=bbox_props, weight='bold',color='w')

    plt.show()    
    
def demo_eigvals_eigvecs_1d(A,pt,figsize=(5,5),ax=None):
    D,U=eig(A)
    #print(f'eigenvalue:\n{D}\neigenvector:\n{U}')       
    ept=np.matmul(A,pt)
    #print(ept)
    
    if ax:
        ax=ax
    else:
        fig, ax = plt.subplots(figsize=figsize)
    

    
    ax.arrow(0,0,U[0][0],U[0][1],width=0.2,head_length=0.1,color='orangered')
    ax.arrow(0,0,U[1][0],U[1][1],width=0.2,head_length=0.1,color='dodgerblue')
    ax.arrow(0,0,ept[0],ept[1],width=0.1,head_length=0.1,color='k')
    ax.arrow(0,0,pt[0],pt[1],width=0.01,head_length=0.1,color='gray')
    
    ax.plot([0,pt[0]],[pt[1],pt[1]],color='gray',linestyle='--')
    ax.plot([pt[0],pt[0]],[0,pt[1]],color='gray',linestyle='--')
    ax.plot([0,ept[0]],[ept[1],ept[1]],color='k',linestyle='--')
    ax.plot([ept[0],ept[0]],[0,ept[1]],color='k',linestyle='--')
    
    bbox_props = dict(boxstyle="round", fc="k", ec="0.5", alpha=0.8)
    ax.text(-0.3, pt[1], "y", ha="center", va="center", size=15,weight='bold',color='gray')
    ax.text(pt[0],-0.3, "x", ha="center", va="center", size=15,weight='bold',color='gray')
    ax.text(-0.6, ept[1], r"$\lambda$y", ha="center", va="center", size=15,weight='bold',color='k')
    ax.text(ept[0],-0.6, r"$\lambda$x", ha="center", va="center", size=15,weight='bold',color='k')    
    ax.text(pt[0]*(2/3),pt[1]*(2/3), r"$x$", ha="center", va="center", size=15,weight='bold',color='w',bbox=bbox_props)  
    ax.text(ept[0]*(2/3),ept[1]*(2/3), r"$Ax=\lambda x$", ha="center", va="center", size=15,weight='bold',color='w',bbox=bbox_props)  
    ax.text(max([ept[0],pt[0],0]),min([ept[1],pt[1],0])-1, f'A:\n{A}\neigenvalue:\n{D}\neigenvector:\n{U}\nscaled vector:\n{ept}', size=10,weight='light',color='w',verticalalignment='top',horizontalalignment='right',bbox=bbox_props)  
    
    #ax.yaxis.set_ticks(np.arange(min([ept[1],pt[1],0]),max([ept[1],pt[1],0]),1))
    ax.scatter(pt[0],pt[1],c='gray',s=100)
    ax.scatter(ept[0],ept[1],c='r',s=100)    
    
    ax.hlines(y=0, xmin=min([ept[0],pt[0],0]), xmax=max([ept[0],pt[0],0]), color='silver',linewidth=3.5)
    ax.vlines(x=0, ymin=min([ept[1],pt[1],0]), ymax=max([ept[1],pt[1],0]), color='silver',linewidth=3.5)     
    if ax:
        pass
    else:
        plt.show()

def demo_KPCA_2d_to_3d_mapping(figsize=(8, 4)):
    X,y=make_circles(n_samples=500,factor=0.5, noise=0.05)
    X+=1.1

    fig=plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection='3d')

    ax1.scatter(X[y==0, 0], X[y==0, 1],color='orangered', marker='o', alpha=0.5)
    ax1.scatter(X[y==1, 0], X[y==1, 1],color='dodgerblue', marker='o', alpha=0.5)

    x1=X[:,0]**2
    y1=X[:,1]**2
    z1=np.sqrt(2*np.prod(X, axis=1))
    ax2.scatter(x1[y==0],y1[y==0],z1[y==0],color='orangered', marker='o', alpha=0.5)
    ax2.scatter(x1[y==1],y1[y==1],z1[y==1],color='dodgerblue', marker='o', alpha=0.5)

    # Set zoom and angle view
    ax2.view_init(0, -45, 0)
    ax2.set_box_aspect(None, zoom=1.1)            

    plt.tight_layout()
    plt.show()    
    
class Demo_Isomap:
    def __init__(self,**kwargs):
        self.args=dict(
            n_samples = 1000,
            noise = 0.05,
            n_neighbors=10,
            n_clusters=6,
            n_components=2,
            source_id=None,
            target_id=None
        )    

        self.args.update(kwargs)        
        
        self.X, _ = make_swiss_roll(self.args['n_samples'], noise=self.args['noise'])
        self.X[:, 1] *=0.5 
        
    def agglomerative_clustering(self):
        connectivity = kneighbors_graph(self.X, n_neighbors=self.args['n_neighbors'], include_self=False)
        print("Compute structured hierarchical clustering...")
        st = time.time()
        ward = AgglomerativeClustering(
            n_clusters=self.args['n_clusters'], connectivity=connectivity, linkage="ward"
            ).fit(self.X)
        elapsed_time = time.time() - st
        self.label = ward.labels_
        print(f"Elapsed time: {elapsed_time:.2f}s")
        print(f"Number of points: {self.label.size}")      
        
    def isomap_G(self):
        self.isomap = manifold.Isomap(n_neighbors=self.args['n_neighbors'], n_components=self.args['n_components'])
        self.S_isomap = self.isomap.fit_transform(self.X)
        
        nbg=self.isomap.nbrs_.kneighbors_graph()
        self.G=nx.from_scipy_sparse_array(nbg)
        self.edge_xyz = np.array([(self.X[u], self.X[v]) for u, v in self.G.edges()])
        
        if self.args['source_id']:
            self.source_id=self.args['source_id']
        else:
            self.source_id=0
        if self.args['target_id']:
            self.target_id=self.args['target_id']
        else:
            self.target_id=np.argmax(self.X[:,2])
        
        self.s_t_path=nx.shortest_path(self.G, source=self.source_id, target=self.target_id,weight='weight')
        self.s_t_3d_coordis=self.X[self.s_t_path]
        
        self.s_t_2d_coordis=self.S_isomap[self.s_t_path]
        self.knn=self.isomap.nbrs_.kneighbors()[1]
        
    def vals4color_cmap(self,vals,cmap='hot',vmin=0, vmax=50):    
        cmap = mpl.colormaps[cmap]
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

        color=cmap(norm(vals))
        return color        
        
    def plot_3d(self,**kwargs):
        args=dict(figsize=(10,10),
                  cmap='Set1',
                  pts_s=60,
                  pts_edgecolor="k",
                  edge_color="darkgray",
                  edge_linewidth=1,
                  path_linewidth=5,
                  path_color='r',
                  source_s=500,
                  source_edgecolor="k",
                  source_color='k',
                  target_s=500,
                  target_edgecolor="k",
                  target_color='k',
                  elev=20, 
                  azim=-90, 
                  roll=0,
                  zoom=1.1
                 )
        args.update(kwargs) 
        
        color=self.vals4color_cmap(self.label,cmap=args['cmap'],vmin=min(self.label),vmax=max(self.label))

        fig = plt.figure(figsize=args['figsize'])
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(*self.X.T, ec="w", s=args['pts_s'],edgecolor=args['pts_edgecolor'],color=color,)

        # Plot the edges
        for vizedge in self.edge_xyz:
            ax.plot(*vizedge.T, color=args['edge_color'],linewidth=args['edge_linewidth'])

        ax.plot(*self.s_t_3d_coordis.T,linewidth=args['path_linewidth'],color=args['path_color'])
        ax.scatter(*self.X[self.source_id],s=args['source_s'],edgecolor=args['source_edgecolor'],color=args['source_color'])
        ax.scatter(*self.X[self.target_id],s=args['target_s'],edgecolor=args['target_edgecolor'],color=args['target_color'])

        def _format_axes(ax):
            """Visualization options for the 3D axes."""
            # Turn gridlines off
            ax.grid(False)
            # Suppress tick labels
            for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
                dim.set_ticks([])
            # Set axes labels
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
        _format_axes(ax)

        # Set zoom and angle view
        ax.view_init(args['elev'], args['azim'], args['roll'])
        ax.set_box_aspect(None, zoom=args['zoom']) 

        fig.tight_layout()
        plt.show()      
        
    def plot_2d(self,**kwargs):
        args=dict(figsize=(20,5),
                  cmap='Set1',
                  edge_color='gray',
                  edge_linewidth=1,
                  edge_alpha=0.5,
                  pts_s=70,
                  pts_edgecolor="k",
                  path_linewidth=5,
                  path_color='r',
                  path_alpha=0.5,
                  source_s=500,
                  source_edgecolor="k",
                  source_color='k',
                  target_s=500,
                  target_edgecolor="k",
                  target_color='k',
                  elev=20, 
                  azim=-90, 
                  roll=0,
                  zoom=1.1                  
                 )
        args.update(kwargs)
        
        color=self.vals4color_cmap(self.label,cmap=args['cmap'],vmin=min(self.label),vmax=max(self.label))
        
        fig,ax=plt.subplots(figsize=args['figsize'])

        # plot lines connecting the same neighboring points from our original data
        for i in range(len(self.X)):
            neighbors = self.knn[i]
            for j in range(len(neighbors)):
                ax.plot(self.S_isomap[[i, neighbors.astype('int')[j]], 0], 
                        self.S_isomap[[i, neighbors.astype('int')[j]], 1], color=args['edge_color'],linewidth=args['edge_linewidth'],alpha=args['edge_alpha']);
                
        ax.scatter(*self.S_isomap.T, ec="w", s=args['pts_s'],edgecolor=args['pts_edgecolor'],color=color,) 

        ax.plot(*self.s_t_2d_coordis.T,linewidth=args['path_linewidth'],color=args['path_color'],alpha=args['path_alpha'])
        ax.scatter(*self.S_isomap[self.source_id],s=args['source_s'],edgecolor=args['source_edgecolor'],color=args['source_color'])
        ax.scatter(*self.S_isomap[self.target_id],s=args['target_s'],edgecolor=args['target_edgecolor'],color=args['target_color'])

        plt.show()        
        
# ref: Using Lagrange multipliers in optimization:  https://kitchingroup.cheme.cmu.edu/blog/2013/02/03/Using-Lagrange-multipliers-in-optimization/       
def demo_lagrange_multiplier_xy3D(**kwargs):    
    args=dict(
          figsize=(5,5),
          elev=20, 
          azim=-90, 
          roll=0,
          zoom=1.1,
          s=150,
          fontsize=10,
        )
    args.update(kwargs)
    
    x = np.linspace(-1.5, 1.5)
    [X, Y] = np.meshgrid(x, x)
    theta = np.linspace(0,2*np.pi);
    R = 1.0
    x1 = R * np.cos(theta)
    y1 = R * np.sin(theta)
    
    def func(X):
        x = X[0]
        y = X[1]
        L = X[2] # this is the multiplier. lambda is a reserved keyword in python
        return x + y + L * (x**2 + y**2 - 1)    

    def dfunc(X):
        # Rather than perform the analytical differentiation, here we develop a way to numerically approximate the partial derivatives.
        dLambda = np.zeros(len(X))
        h = 1e-3 # this is the step size used in the finite difference.
        for i in range(len(X)):
            dX = np.zeros(len(X))
            dX[i] = h
            dLambda[i] = (func(X+dX)-func(X-dX))/(2*h);
        return dLambda    
    
    X1 = fsolve(dfunc, [1, 1, 0])
    X2 = fsolve(dfunc, [-1, -1, 0])
                
    fig = plt.figure(figsize=args['figsize'])
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, X + Y,cmap=cm.coolwarm,alpha=0.5)    
                
    ax.scatter(*X1[:2],func(X1),s=args['s'],color='r',marker='o')
    ax.scatter(*X2[:2],func(X2),s=args['s'],color='r',marker='o') 
    
    ax.plot(x1, y1, x1 + y1, 'r--')
    
    # Demo 2: color
    ax.text(*X1[:2],func(X1),s=f'({X1[0]:.3f},{X1[1]:.3f},{X1[2]:.3f});{func(X1):.3f}', color='red',fontsize=args['fontsize'],ha='right')
    ax.text(*X2[:2],func(X2),s=f'({X2[0]:.3f},{X2[1]:.3f},{X2[2]:.3f});{func(X2):.3f}', color='red',fontsize=args['fontsize'],ha='left')
    
    # Set zoom and angle view
    ax.view_init(args['elev'], args['azim'], args['roll'])
    ax.set_box_aspect(None, zoom=args['zoom']) 

    fig.tight_layout()
    plt.show()          
        