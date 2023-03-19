# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 19:37:03 2023

@author: richie bao
"""
import numpy  as np
import os
from collections import Counter

from usda.pattern_signature import  _grid_neighbors_xy_finder as nghxy_finder
# from ..pattern_signature import  _grid_neighbors_xy_finder as nghxy_finder

def target_function(quadrat):
    global compared_quadradt    

    q1_cc=cc3d.connected_components(quadrat,connectivity=8,return_N=False,out_dtype=np.uint64)
    q1_cs=usda_signature.class_clumpSize_histogram(quadrat,q1_cc)   
    
    q2_cc=cc3d.connected_components(compared_quadradt, connectivity=8,return_N=False,out_dtype=np.uint64)
    q2_cs=usda_signature.class_clumpSize_histogram(compared_quadradt,q2_cc)       
    
    q1_cs_pdf=q1_cs/q1_cs.values.sum()
    q2_cs_pdf=q2_cs/q2_cs.values.sum()

    q1_cs_pdf,q2_cs_pdf=usda_utils.complete_dataframe_rowcols([q1_cs_pdf,q2_cs_pdf])     
    
    class_clumpSize_pdf_shannon=usda_signature.Distances(q1_cs_pdf.to_numpy().flatten(),q2_cs_pdf.to_numpy().flatten())
    distance=class_clumpSize_pdf_shannon.shannon()['Jensen-Shan']

    return distance

# Function
# def target_function():
#     return

def initial_position(object_idx,swarm_size=5, rows_n=5, cols_n=5,target_function=target_function):
    position=np.random.choice(object_idx,(swarm_size, rows_n,cols_n))
    target=np.array(list(map(target_function,position)))
    
    return position, target

def initial_velocity(position,objects_idx):   
    velocity=np.random.random(list(position.shape)+[len(objects_idx)])
    row_sums = velocity.sum(axis=-1)
    nomalized_velocity=velocity/row_sums[:,:,:,np.newaxis]
    
    return nomalized_velocity

def update_position(velocity,objects_idx,target_function=target_function):    
    def update_cell(v):    
        new_position=objects_idx[v.argsort()[-1]]
        
        return new_position
   
    vfunc=np.vectorize(update_cell,signature='(i)->()')
    updated_position=vfunc(velocity)    
    updated_target=np.array(list(map(target_function,updated_position)))
    
    return updated_position,updated_target

def individual_best_matrix(position, i_b_matrix,target,updated_target): 
    for i in range(0, position.shape[0]):
        if target[i]>updated_target[i]:
            i_b_matrix[i]=position[i]
            target[i]=updated_target[i]
            
    return i_b_matrix,target

def velocity_vector(position, init_velocity, i_b_matrix, best_global,objects_idx,c,w,ngh_w):    
    def velocity_cell(p,i_b,b_g,v):    
        r1=int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
        r2=int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)

        v_dict=dict(zip(objects_idx,v))
        v_dict_updated=copy.deepcopy(v_dict)
        pool_idx=list(set(objects_idx)-set([p,i_b,b_g]))
        pool=[v_dict[i] for i in pool_idx]+[round(v_dict[p]*(1-w),3)]
        pool_sum=sum(pool)
        global_bonus=pool_sum*c*r1
        local_bonus=pool_sum*(1-c)*r2
        remaining_bonus=pool_sum-(global_bonus+local_bonus)
        
        if i_b==b_g:
            v_dict_updated.update({i_b:v_dict[i_b]+local_bonus+global_bonus})
        else:
            v_dict_updated.update({i_b:v_dict[i_b]+local_bonus,b_g:v_dict[b_g]+global_bonus})   
        
        pool_idx_remaining_bonus={i:remaining_bonus*(v_dict[i]/(sum([v_dict[j] for j in pool_idx]))) for i in pool_idx}
        if p in [i_b,b_g]:
            v_dict_updated[p]=v_dict_updated[p]-v_dict[p]*(1-w)
            v_dict_updated.update(pool_idx_remaining_bonus)      
        else:
            v_dict_updated[p]=v_dict[p]*w
            v_dict_updated.update(pool_idx_remaining_bonus)
   
        return np.array(list(v_dict_updated.values()))   
        
    vfunc=np.vectorize(velocity_cell,signature='(),(),(),(i)->(i)')     
    updated_init_velocity=vfunc(position, i_b_matrix, best_global,init_velocity)
    
    def velocity_nghs(p,v,idx):
        # print(p,v,idx)   
        nghs=ngh_finder.find(idx[0],idx[1])
        ngh_vals=[particle[j[0],j[1]] for j in nghs]
        ngh_vals.remove(p)
        fre=Counter(ngh_vals)
        nghs_fre_dict={k:0 for k in objects_idx}
        nghs_fre_dict.update(fre)
        nghs_prob_dict={k:v/sum(nghs_fre_dict.values()) for k,v in nghs_fre_dict.items()}
        nghs_prob_array=np.array(list(nghs_prob_dict.values()))
        velocity_nghs=nghs_prob_array*ngh_w+v*(1-ngh_w)
        
        return velocity_nghs

    x_=np.linspace(0, position.shape[1]-1, position.shape[1])
    y_=np.linspace(0, position.shape[2]-1, position.shape[2])
    x_idx, y_idx=np.meshgrid(x_, y_)  
    xy=np.stack((y_idx,x_idx),axis=2).reshape(position.shape[1],position.shape[2],2).astype(int)
    ngh_finder=nghxy_finder.GridNghFinder(0, 0, position.shape[1]-1,position.shape[2]-1)
    
    velocity_nghs_lst=[]
    for i in range(position.shape[0]):       
        vfunc_nghs=np.vectorize(velocity_nghs,signature='(),(i),(n)->(i)')   
        particle=position[i]
        updated_init_velocity_nghs=vfunc_nghs(particle,updated_init_velocity[i],xy)    
        velocity_nghs_lst.append(updated_init_velocity_nghs)

    return np.array(velocity_nghs_lst)

def particle_swarm_optimization_2d(objects_idx,rows_n=5,cols_n=5,swarm_size=5,c=0.5,w=0.9,ngh_w=0.5,iterations = 50,target_function= target_function, verbose=1):
    count=0
    
    position,target=initial_position(objects_idx,swarm_size,rows_n,cols_n,target_function)    
    init_velocity=initial_velocity(position,objects_idx)    

    i_b_matrix=np.copy(position)
    sorted_idx=target.argsort()
    best_global=np.copy(position[sorted_idx][0,:])    
    best_target=target[sorted_idx[0]]    
    
    epoch={}
    while (count <= iterations):
        if count%verbose==0:     
            print('Iteration = ', count, ' f(x) = ', best_target)   
        
        epoch[count]=best_target
   
        position,updated_target=update_position(init_velocity,objects_idx,target_function)          
        i_b_matrix,target=individual_best_matrix(position, i_b_matrix,target,updated_target)                    
        
        sorted_idx=target.argsort()
        value=target[sorted_idx[0]]
        if best_target>value:
            best_target=value
            best_global=np.copy(position[sorted_idx][0,:])   
        
        init_velocity=velocity_vector(position, init_velocity, i_b_matrix, best_global,objects_idx,c,w,ngh_w)

        count+=1   
        # break
    return best_global,epoch

if __name__=="__main__":
    from usda import datasets as usda_datasets
    from usda import data_visualization as usda_vis
    from usda import pattern_signature as usda_signature
    from usda import utils as usda_utils
    
    import mapclassify
    import matplotlib
    import cc3d
    import copy
    import matplotlib.pyplot as plt
    
    lu_class_idNcolor={
    'intensive_agriculture':[1,"#ca9146"],
    'extensive_agriculture':[2,"#ebcb90"],
    'residence':[3,"#b50000"],
    'industrye':[4,"#e29e8c"],
    'recreation_day_tripsy':[5,"#38814e"],
    'recreation_overnight':[6,"#d4e7b0"],
    'wet_natural_area':[7,"#c8e6f8"],
    'water_recreational_use':[8,"#64b3d5"],
    'water_limited_access':[9,'#5475a8'],
    }
    
    lu_class_color={v[0]:v[1] for v in lu_class_idNcolor.values()}
    cmap_LC, norm=matplotlib.colors.from_levels_and_colors(list(lu_class_color.keys()),list(lu_class_color.values()),extend='max')
    cmap_LC
    
    n=16
    size=n
    X,_=usda_datasets.generate_categorical_2darray(size=size,seed=99)
    X4=mapclassify.FisherJenks(X[0], k=4).yb.reshape(size,size)+1
    usda_vis.imshow_label2darray(X4,figsize=(20,20),fontsize=10,cmap=cmap_LC,norm=norm)      
    
    objects_idx=list(range(1,5))
    rows_n=n
    cols_n=n
    compared_quadradt=copy.deepcopy(X4)
    
    pattern,epoch=particle_swarm_optimization_2d(
        objects_idx,
        rows_n=rows_n,
        cols_n=cols_n,
        c=0.5,
        w=0.9,
        ngh_w=0.5, # 0.5,0.2
        swarm_size=20,
        target_function=target_function, 
        iterations=10,
        verbose=1)    
    
    usda_vis.imshow_label2darray(pattern,figsize=(20,20),fontsize=10,cmap=cmap_LC,norm=norm)    
    
    # fig, ax=plt.subplots()
    # ax.plot(epoch.keys(),epoch.values())
    # plt.show()