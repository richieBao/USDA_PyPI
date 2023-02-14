# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 15:58:40 2023

@author: richie bao
"""
from ..pattern_signature import  _signature as usda_signature
from ..utils import _df_process as usda_df_process
from ..pattern_signature import _distance_metric as usda_distance

import cc3d
import numpy as np
import itertools
import pandas as pd

def signature2distance_integrating(scenes,signatures_lst=['class_hierarchical_decomposition']):
    '''
    批量计算signature（标记特征），并匹配siganature选择distance（距离）算法

    Parameters
    ----------
    scenes : list[2darray]
        多个2维数组（代表栅格或图像）.
    signatures_lst : list[str], optional
        signature名称列表，目前含有class_clumpSize、class_pairs_frequency和class_hierarchical_decomposition. The default is ['class_hierarchical_decomposition'].

    Returns
    -------
    pattern_distance_df : DataFrame
        signature标记特征对应distance距离返回值.

    '''
    scene_keys,scene_vals=scenes.keys(),scenes.values()    
    class_clumpSize_func=lambda v: usda_signature.class_clumpSize_histogram(v.data,cc3d.connected_components(v.data,connectivity=8,return_N=False,out_dtype=np.uint64) )
    class_pairs_frequency_func=lambda v: usda_signature.class_co_occurrence(v.data)
    class_hierarchical_decomposition_func=lambda v: usda_signature.class_decomposition(v.data)
    
    signature_func_dict={'class_clumpSize':class_clumpSize_func,
                         'class_pairs_frequency':class_pairs_frequency_func,
                         'class_hierarchical_decomposition':class_hierarchical_decomposition_func}   
    
    
    distance_func_dict={'class_clumpSize':lambda instance:instance.shannon()['Jensen-Shan'],
                        'class_pairs_frequency':lambda instance:instance.intersection()['Wave Hedges'],
                        'class_hierarchical_decomposition':lambda instance:instance.inner()['Jaccard']}   
    
    
    signatures=[list(map(signature_func_dict[sig],scene_vals)) for sig in signatures_lst]
    signatures_T=list(zip(*signatures))
    signatures_dict={k:{k_:v_ for k_,v_ in zip(signatures_lst,v)} for k,v in zip(scene_keys,signatures_T)}
    
    signatures_pdf={k:{k_:v_/v_.values.sum() for k_,v_ in v.items()} for k,v in signatures_dict.items()}
    scene_idx_pairs=list(itertools.combinations(scene_keys,2))
        
    pattern_distance={k:{} for k in signatures_lst}
    for p_a,p_b in scene_idx_pairs:        
        sig_pa_=signatures_pdf[p_a]  
        sig_pb_=signatures_pdf[p_b]
        
        for sig in signatures_lst:
            sig_pa=sig_pa_[sig]
            sig_pb=sig_pb_[sig]
            
            sig_pa,sig_pb=usda_df_process.complete_dataframe_rowcols([sig_pa,sig_pb])
            distance_instance=usda_distance.Distances(sig_pa.to_numpy().flatten(),sig_pb.to_numpy().flatten())
            distance=list(map(distance_func_dict[sig],[distance_instance]))[0]

            pattern_distance[sig][(p_a,p_b)]=distance
            
    pattern_distance_df=pd.DataFrame(pattern_distance)
    
    return pattern_distance_df