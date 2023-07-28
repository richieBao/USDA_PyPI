# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 15:11:00 2022

@author: Richie Bao-caDesign设计(cadesign.cn)
"""
from ._manifold_learning_illustration import plot_embedding
from ._manifold_learning_illustration import manifold_learning_illustration
from ._rbf_kernel_pca import rbf_kernel_pca
from ._correlations_embedding import partial_correlations_embedding2Dgraph
from ._tsne import Hbeta
from ._tsne import x2p
from ._tsne import pca
from ._tsne import tsne

__all__=[
    "manifold_learning_illustration",
    "plot_embedding",
    "rbf_kernel_pca",
    "partial_correlations_embedding2Dgraph",
    "Hbeta",
    "x2p",
    "pca",
    "tsne",
    ]



