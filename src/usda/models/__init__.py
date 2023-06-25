# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 15:11:00 2022

@author: Richie Bao-caDesign设计(cadesign.cn)
"""
from ._neighbors import k_neighbors_entire
from ._computational_performance import PolynomialFeatures_regularization

from ._dim1_convolution import dim1_convolution_SubplotAnimation
from ._dim1_convolution import G_T_type_1
from ._dim1_convolution import F_T_type_1
from ._dim1_convolution import G_T_type_2
from ._dim1_convolution import F_T_type_2
from ._curve_segmentation import curve_segmentation_1DConvolution
from ._sir_model import SIR_deriv
from ._sir_model import convolution_diffusion_img
from ._sir_model import SIR_spatialPropagating
from ._superpixel_segmentation import superpixel_segmentation_Felzenszwalb
from ._superpixel_segmentation import superpixel_segmentation_quickshift
from ._superpixel_segmentation import multiSegs_stackStatistics
from ._bow_feature_builder import feature_builder_BOW
from ._label_encoder import df_multiColumns_LabelEncoder
from ._label_encoder import text2ints_encoded
from ._entropy import entropy_compomnent
from ._entropy import IG
from ._decision_tree import decisionTree_structure
from ._random_forest_classifier import ERF_trainer
from ._image_tag_extractor import ImageTag_extractor
from ._clustering import clustering_minibatchkmeans_selectkbest_ns
from ._global_local_autocorrelation import moran_local_autocorrelation_gdf

from ._rnn_lstm import RNN_LSTM_sequence
from ._rnn_lstm import RNN_LSTM_train_sequence
from ._rnn_lstm import RNN_model_img
from ._rnn_lstm import RNN_train_img
from ._rnn_lstm import CharModel
from ._rnn_lstm import char_train
from ._rnn_lstm import char_random_generation

from ._nlp import build_co_occurrence_matrix

__all__=[
    "k_neighbors_entire",
    "PolynomialFeatures_regularization",
    "dim1_convolution_SubplotAnimation",
    "G_T_type_1",
    "F_T_type_1",
    "G_T_type_2",
    "F_T_type_2",
    "curve_segmentation_1DConvolution"
    "SIR_deriv",
    "convolution_diffusion_img",
    "superpixel_segmentation_Felzenszwalb",
    "superpixel_segmentation_quickshift",
    "multiSegs_stackStatistics",
    "feature_builder_BOW",
    "df_multiColumns_LabelEncoder",
    "entropy_compomnent",
    "IG",
    "decisionTree_structure",
    "ERF_trainer",
    "ImageTag_extractor",
    "SIR_spatialPropagating",
    "clustering_minibatchkmeans_selectkbest_ns",
    "moran_local_autocorrelation_gdf",
    "RNN_LSTM_sequence",
    "RNN_LSTM_train_sequence",
    "RNN_model_img",
    "RNN_train_img",
    "text2ints_encoded",
    "CharModel",
    "char_train",
    "char_random_generation",
    "build_co_occurrence_matrix",
    ]



