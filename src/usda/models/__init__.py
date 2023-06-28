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

from ._nlp_tools import build_co_occurrence_matrix
from ._nlp_tools import text_replace_preprocess
from ._nlp_tools import create_lookup_tables4vocab
from ._nlp_tools import subsampling_of_frequent_words
from ._nlp_tools import get_batches4word2vec
from ._nlp_tools import get_target4word2vec
from ._nlp_tools import Word2idx_idx2word
from ._nlp_tools import unicodeToAscii
from ._nlp_tools import readLangs2langs
from ._nlp_tools import filterPair
from ._nlp_tools import filterPairs
from ._nlp_tools import prepareData4seq2seq


from ._word2vec_sgns import SkipGramNeg
from ._word2vec_sgns import NegativeSamplingLoss
from ._word2vec_sgns import cosine_similarity
from ._word2vec_sgns import noise_dist4sgns
from ._word2vec_sgns import sgns_train

from ._seq2seq import EncoderRNN
from ._seq2seq import DecoderRNN
from ._seq2seq import BahdanauAttention
from ._seq2seq import AttnDecoderRNN
from ._seq2seq import indexesFromSentence
from ._seq2seq import tensorFromSentence
from ._seq2seq import tensorsFromPair
from ._seq2seq import get_dataloader
from ._seq2seq import seq2seq_train
from ._seq2seq import seq2seq_evaluateRandomly
from ._seq2seq import showAttention
from ._seq2seq import evaluateAndShowAttention


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
    "text_replace_preprocess",
    "create_lookup_tables4vocab",
    "subsampling_of_frequent_words",
    "get_batches4word2vec",
    "SkipGramNeg",
    "NegativeSamplingLoss",
    "cosine_similarity",
    "noise_dist4sgns",
    "sgns_train",
    "get_target4word2vec",
    "Word2idx_idx2word",
    "unicodeToAscii",
    "readLangs2langs",
    "filterPair",
    "filterPairs",
    "prepareData4seq2seq",
    ]



