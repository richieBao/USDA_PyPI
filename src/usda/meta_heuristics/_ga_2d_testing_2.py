# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 15:45:56 2023

@author: richie bao 
"""

from usda import datasets as usda_datasets
from usda import data_visualization as usda_vis
from usda import pattern_signature as usda_signature
from usda import utils as usda_utils

import mapclassify
import cc3d


# Required Libraries
import numpy  as np
import random
import copy
import os
from usda.pattern_signature import  _grid_neighbors_xy_finder as nghxy_finder
# from ..pattern_signature import  _grid_neighbors_xy_finder as nghxy_finder


###############################################################################

from usda import datasets as usda_datasets
from usda import data_visualization as usda_vis
from usda import pattern_signature as usda_signature
from usda import utils as usda_utils
from usda import meta_heuristics as usda_heuristicsw
from usda import network as usda_network

import mapclassify
import numpy  as np
import cc3d
import copy
import matplotlib.pyplot as plt
import random
import itertools
import networkx as nx
from networkx.algorithms import approximation as approx   
import math
from toolz import partition
from collections import defaultdict
from scipy.signal import convolve2d

jisperveld_data=usda_datasets.load_jisperveld_data()
# domain_objectives_worst2best={'natrue_cost':[1*400,10*400],'recreation_cost':[1*400,10*400],'lu_conversion_cost':[300*400,-10000*400],'C':[5*9,1*9],'L':[0.2*9,1*9],'R':[0,1]} # 2818*400,-43155*400

domain_objectives_worst2best={'natrue_cost':[1*400,10*400],'recreation_cost':[1*400,10*400],'lu_conversion_cost':[300*400,-10000*400],'C':[5*9,1*9],'L':[0.2*9,1*9],'R':[1,0]} # 2818*400,-43155*400
domain_lu_area={'intensive_agriculture':[80,150],
                'extensive_agriculture':[20,65],
                'residence':[20,45], 
                'industry':[5,15], 
                'recreation_day_trips':[0,70], 
                'recreation_overnight':[0,35],
                'wet_natural_area':[0,30],
                'water_recreational_use':[120,150],
                'water_limited_access':[0,60]}

def target_function_lu_area(lus):
    global domain_lu_area,jisperveld_data   
    
    lu_name=jisperveld_data['lu_name']
    lu_idx2name={v:k for k,v in lu_name.items()}
    domain_lu_area_idx={lu_idx2name[k]:v for k,v in domain_lu_area.items()}
    # print(domain_lu_area_idx)
    unique, counts=np.unique(lus, return_counts=True)
    unique_counts=dict(zip(unique, counts))
    # print(unique_counts)
    cost=0
    for idx,fre in unique_counts.items():
        domain=domain_lu_area_idx[idx]
        dis=max([max(0,domain[0]-fre),max(0,fre-domain[1])])
        dis_abs_fraction=abs(dis)/lus.size
        cost+=dis_abs_fraction        
        
    return cost   

def target_function_additive_objectives_constrains(lus):
    global jisperveld_data        
    
    nature_recreation_vals=jisperveld_data['nature_recreation_vals']    
    lu_name=jisperveld_data['lu_name']
    # lu2ID={v:k for k,v in lu_name.items()}
    nature_vals_lst=[]
    recreation_vals_lst=[]    
    
    for k,v in lu_name.items():        
        lu_mask=lus==k
        
        # natural cost
        nature_val=nature_recreation_vals['nature_value'][v]        
        if type(nature_val)==str:
            nature_vals_map=jisperveld_data[nature_val]
            lu_nature_val=nature_vals_map*lu_mask   
        else:
            lu_nature_val=lu_mask*nature_val
  
        nature_vals_lst.append(lu_nature_val)
        
        # recreational cost
        recreation_val=nature_recreation_vals['recreational_value'][v]  
        if type(recreation_val)==str:
            recreation_vals_map=jisperveld_data[recreation_val]
            lu_recreation_val=recreation_vals_map*lu_mask   
        else:
            lu_recreation_val=lu_mask*recreation_val
            
        recreation_vals_lst.append(lu_recreation_val)    
        
    # lu conversion cost
    lus_original=jisperveld_data['lu']
    changed_mask=~((lus-lus_original)==0)
    original=lus_original[changed_mask]
    changed=lus[changed_mask]
    original2changed=list(zip(original,changed))
    
    conversion_cost_matrix=jisperveld_data['lu_conversion_cost'].T
    conversion_cost_matrix.fillna(9999999999,inplace=True)
    
    conversion_cost=0
    for pair in original2changed:       
        conversion_pair_cost=conversion_cost_matrix[lu_name[pair[0]]][lu_name[pair[1]]]
        conversion_cost+=conversion_pair_cost           
    
    nature_costs=np.array(nature_vals_lst).sum(axis=0)
    recreation_costs=np.array(recreation_vals_lst).sum(axis=0)
    
    nature_cost=nature_costs.sum()
    recreation_cost=recreation_costs.sum()
    
    # print(conversion_cost)
    return nature_cost,recreation_cost,conversion_cost

# jisperveld_lu_changed=jisperveld_lu+(jisperveld_lu==3)*3 # 测试用假设已变化的土地利用类型
# nature_cost,recreation_cost,conversion_cost=target_function_additive_objectives_constrains(jisperveld_lu_changed)
# print(nature_cost,recreation_cost,conversion_cost)        

def target_function_spatial_objectives_constrains(lus):
        
    #  minimize fragmentation
    clump_2darray,C=cc3d.connected_components(lus,connectivity=8,return_N=True,out_dtype=np.uint64) 
   
    # maximize the largest cluster
    unique, counts=np.unique(clump_2darray, return_counts=True)
    unique_counts=dict(zip(unique, counts))
    class_clump=np.stack((lus,clump_2darray),axis=2)
    class_clump_mapping=usda_signature.lexsort_based(class_clump.reshape(-1,2)).tolist()
    class_clump_mapping.sort(key=lambda x:x[0])
    class_clump_max=[[int(i[0]),unique_counts[i[1]]] for i in class_clump_mapping]
    cluster_num_dict=defaultdict(list)    
    for k,v in class_clump_max:
        cluster_num_dict[k].append(v)
    L_k={k:max(v)/sum(v) for k,v in cluster_num_dict.items()}
    L=sum(L_k.values())

    # 由簇边长除以簇数量的均值，调整为簇边缘栅格单元数与栅格单元总数的比例。簇边缘栅格单元的确定方法使用了卷积边缘检测的方式。
    #  maximize compactness
    # kernel_edge=np.array([[-1,-1,-1],
    #                       [-1,8,-1],
    #                       [-1,-1,-1]]) 
    # edges=convolve2d(clump_2darray,kernel_edge,mode='same')
    # R=(edges!=0).sum()/lus.size
    # print('-'*50)
    xextent,yextent=lus.shape
    ngh_finder=nghxy_finder.GridNghFinder(0, 0, xextent-1,yextent-1)
    x_=np.linspace(0, xextent-1, xextent)
    y_=np.linspace(0, yextent-1, yextent)
    x_idx, y_idx=np.meshgrid(x_, y_) 
    xy=np.stack((x_idx,y_idx),axis=2).reshape(-1,2).astype(int)
    pairs=np.empty((0,2),int)
    R=0
    for i in xy:
        nghs=ngh_finder.find(i[0],i[1])
        ngh_vals=[lus[j[0],j[1]] for j in nghs]
        i_val=lus[i[0],i[1]]
        ngh_vals.remove(i_val)
        i_pairs=np.array([[i_val,k] for k in ngh_vals])
        i_pairs_set=set(list(zip(i_pairs.T[0],i_pairs.T[1])))
        # print(i_pairs, i_pairs_set)        
        pairs_num_fraction=len(i_pairs_set)/8
        # print(pairs_num_fraction)
        R+=pairs_num_fraction/400
    
    return C,L,R
    
# C,L,R=target_function_spatial_objectives_constrains(jisperveld_lu)    
# print(C,L,R)    



def target_function(lus):
    global domain_objectives_worst2best,cost_filter

    nature_cost,recreation_cost,conversion_cost=target_function_additive_objectives_constrains(lus)
    C,L,R=target_function_spatial_objectives_constrains(lus)     
    
    cost_dict_={'natrue_cost':nature_cost,'recreation_cost':recreation_cost,'lu_conversion_cost':conversion_cost,'C':C,'L':L,'R':R,'lu_area_cost':None}     
    cost_dict={k:v for k,v in cost_dict_.items() if k in cost_filter}
    
    p=4
    priority_level=0.5
    cost=0
    
    for k,v in cost_dict.items():
        if k=='lu_area_cost':
            lu_area_cost=target_function_lu_area(lus)    
            cost+=lu_area_cost        
        else:
            domain=domain_objectives_worst2best[k]        
            target_val=(domain[1]-domain[0])*priority_level 
            g_v=(v-domain[1])/(target_val-domain[1])

            g_v_power=pow(g_v,p)
            cost+=g_v_power
     
    return cost
    
# cost=target_function(jisperveld_lu_changed)
# print(cost)  

# Function
# def target_function():
#     return


def population_replace_fixed_map(population,fixed_map):
    pop_lst=[]
    for chromosome in np.copy(population):
        chromosome[fixed_map!=0]=fixed_map[fixed_map!=0]
        pop_lst.append(chromosome)
        
    return np.array(pop_lst)

# Function: Initialize Variables
# def initial_population(object_idx,population_size=5, rows_n=5, cols_n=5,target_function=target_function,fixed_map=None):
#     population=np.random.choice(object_idx,(population_size, rows_n,cols_n))
#     if fixed_map is not None:        
#         population=population_replace_fixed_map(population,fixed_map)
        
        
#     target=np.array(list(map(target_function,population)))

#     return population, target

def initial_population(object_idx,population_size=5, rows_n=5, cols_n=5,target_function=target_function,fixed_map=None,population_init=None):
    global jisperveld_data  
        
    # population=np.random.choice(object_idx,(population_size, rows_n,cols_n))
    if population_init is not None:
        # print('999999999999')
        population=np.array([population_init]*population_size)
    else:
        # print('888888888888888888')
        population=np.random.choice(object_idx,(population_size, rows_n,cols_n))
    
    if fixed_map is not None:        
        population=population_replace_fixed_map(population,fixed_map)        
        
    target=np.array(list(map(target_function,population)))

    # print(population)

    return population, target

# Function: Fitness
def fitness_function(target): 
    fitness=np.zeros((target.shape[0], 2))
    for i in range(0, fitness.shape[0]):
        fitness[i,0]=1/(1+ target[i]+ abs(min(target)))

    fit_sum=fitness[:,0].sum()
    fitness[0,1]=fitness[0,0]
    for i in range(1, fitness.shape[0]):
        fitness[i,1]=(fitness[i,0] + fitness[i-1,1])
    for i in range(0, fitness.shape[0]):
        fitness[i,1]=fitness[i,1]/fit_sum
    
    return fitness

# Function: Selection
def roulette_wheel(fitness): 
    ix=0
    random=int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
    for i in range(0, fitness.shape[0]):
        if (random <= fitness[i, 1]):
          ix=i
          break
    return ix

def crossover_tsai(p_1,p_2):
    # print(p_1,'\n','-'*50,'\n',p_2)
    rand_r=random.randint(0,p_1.shape[0])
    rand_c=random.randint(0,p_1.shape[1])
    rand_a= int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1) 
    rand_b= int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1) 
    # print(rand_r,rand_c,rand_a,rand_b)
    
    offspring_individual_1=np.copy(p_1)
    offspring_individual_2=np.copy(p_2)
    offspring_individual_3=np.copy(p_1)
    offspring_individual_4=np.copy(p_2)   

    idx_along_row=[(i,j) for i in range(p_1.shape[0]) for j in range(p_1.shape[1])]
    idx_along_col=[(i,j) for j in range(p_1.shape[1]) for i in range(p_1.shape[0])]
    # print(idx_along_row,idx_along_col)
    
    if rand_a>0.5:     
        idx=idx_along_row[:rand_r*p_1.shape[0]+rand_c]
        # idx_b=idx_along_row[rand_r*p_1.shape[0]+rand_c:]
        for i in idx:
            offspring_individual_1[i]=p_2[i]
            offspring_individual_2[i]=p_1[i]
    else:
        idx=idx_along_col[:rand_r*p_1.shape[0]+rand_c]
        # idx_b=idx_along_row[rand_r*p_1.shape[0]+rand_c:]
        for i in idx:
            offspring_individual_3[i]=p_2[i]
            offspring_individual_4[i]=p_1[i]      
    choice_idx=np.random.choice([0,1,2,3])
    offspring_individuals=[offspring_individual_1,offspring_individual_2,offspring_individual_3,offspring_individual_4]
    offspring_individual=offspring_individuals[choice_idx]
    target=target_function(offspring_individual)        
    
    return offspring_individual,target

def crossover_CBO(p_1,p_2):
    # print(p_1)
    # print(p_2)
    rand_r_1=random.randint(0,p_1.shape[0]-1)
    rand_c_1=random.randint(0,p_1.shape[1]-1)    
    rand_r_2=random.randint(0,p_1.shape[0]-1)
    rand_c_2=random.randint(0,p_1.shape[1]-1) 
    
    # print(rand_r_1,rand_c_1,rand_r_2,rand_c_2)
    
    offspring_individual=np.copy(p_1)
    
    ngh_finder=nghxy_finder.GridNghFinder(0, 0, p_1.shape[0]-1,p_1.shape[1]-1)
    nghs=ngh_finder.find(rand_r_1,rand_c_1)                                                
    ngh_vals=[p_1[j[0],j[1]] for j in nghs]   
    i_val=p_1[rand_r_1,rand_c_1]
    ngh_vals.remove(i_val)   
    
    p_2_gene=p_2[rand_r_2,rand_c_2]
    
    while p_2_gene not in ngh_vals and i_val!=p_2_gene:
        rand_r_1=random.randint(0,p_1.shape[0]-1)
        rand_c_1=random.randint(0,p_1.shape[1]-1)    
        rand_r_2=random.randint(0,p_1.shape[0]-1)
        rand_c_2=random.randint(0,p_1.shape[1]-1) 
        
        ngh_finder=nghxy_finder.GridNghFinder(0, 0, p_1.shape[0]-1,p_1.shape[1]-1)
        nghs=ngh_finder.find(rand_r_1,rand_c_1)                                                
        ngh_vals=[p_1[j[0],j[1]] for j in nghs]   
        i_val=p_1[rand_r_1,rand_c_1]
        ngh_vals.remove(i_val)   
        
        p_2_gene=p_2[rand_r_2,rand_c_2]        
    
    offspring_individual[rand_r_1,rand_c_1]=p_2_gene
    target=target_function(offspring_individual)    
    
    return offspring_individual,target    
        
# Function: Offspring
def breeding(population,target, fitness,crossover_name='crossover_CBO', elite=0, target_function=target_function,fixed_map=None):
    offspring=np.copy(population)
    # print(offspring)
    b_offspring=0
    if (elite>0):
        preserve=np.copy(population[target.argsort()])
        for i in range(0, elite):
            offspring[i]=preserve[i]
    target_updated=[]       
    for i in range(elite,offspring.shape[0]):
        parent_1, parent_2=roulette_wheel(fitness), roulette_wheel(fitness)
        while parent_1==parent_2:
            parent_2=random.sample(range(0, len(population) - 1), 1)[0]
        
        p_1=offspring[parent_1]
        p_2=offspring[parent_2]   
        
        if crossover_name=='crossover_tsai':
            offspring_individual,target=crossover_tsai(p_1,p_2)            
        elif crossover_name=='crossover_CBO':
            offspring_individual,target=crossover_CBO(p_1,p_2)        
        
        if fixed_map is not None:
            offspring_individual[fixed_map!=0]=fixed_map[fixed_map!=0]
            
        offspring[i]=offspring_individual
        target_updated.append(target)
        
        # break
    
    return offspring,target_updated

# Function: Mutation        
def mutation_tsai_1(offspring,target_updated, mutation_rate=0.1, target_function = target_function,fixed_map=None):
    for i in range (0, offspring.shape[0]):
        target=target_updated[i]
        chromosome=offspring[i]
        # print(chromosome.shape)
        # print(chromosome)
        probability=int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
        # print(probability)
        if (probability < mutation_rate):
            rand_r_1=random.randint(0,chromosome.shape[0]-1)
            rand_c_1=random.randint(0,chromosome.shape[1]-1)    
            rand_r_2=random.randint(0,chromosome.shape[0]-1)
            rand_c_2=random.randint(0,chromosome.shape[1]-1) 

            while rand_r_1==rand_r_2:
                rand_r_2=random.randint(0,chromosome.shape[0]-1)
            while rand_c_1==rand_c_2:
                rand_c_2=random.randint(0,chromosome.shape[1]-1) 
            # print(rand_r_1,rand_r_2,rand_c_1,rand_c_2)                
            gene_1=chromosome[(rand_r_1,rand_c_1)]
            gene_2=chromosome[(rand_r_2,rand_c_2)]
            chromosome[(rand_r_1,rand_c_1)]=gene_2
            chromosome[(rand_r_2,rand_c_2)]=gene_1
            
            if fixed_map is not None:
                chromosome[fixed_map!=0]=fixed_map[fixed_map!=0]
            
            target=target_function(chromosome)            
            
        if fixed_map is not None:
            chromosome[fixed_map!=0]=fixed_map[fixed_map!=0]  
            
        offspring[i]=chromosome
        target_updated[i]=target
    
    return offspring,np.array(target_updated)    

# Function: Mutation        
def mutation_tsai_2(offspring,target_updated, mutation_rate=0.1, target_function = target_function,fixed_map=None):
    for i in range (0, offspring.shape[0]):
        target=target_updated[i]
        chromosome=offspring[i]
        # print(chromosome.shape)
        # print(chromosome)
        probability=int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
        # print(probability)
        
        rand_r_1=random.randint(0,chromosome.shape[0]-1)
        rand_c_1=random.randint(0,chromosome.shape[1]-1)    
        rand_r_2=random.randint(0,chromosome.shape[0]-1)
        rand_c_2=random.randint(0,chromosome.shape[1]-1)          
        
        while rand_r_1==rand_r_2:
            rand_r_2=random.randint(0,chromosome.shape[0]-1)
        while rand_c_1==rand_c_2:
            rand_c_2=random.randint(0,chromosome.shape[1]-1)   
        
        # print(rand_r_1,rand_r_2,rand_c_1,rand_c_2)
        if (probability > mutation_rate):
            chromosome[[rand_r_1,rand_r_2]]=chromosome[[rand_r_2,rand_r_1]]

        else:
            chromosome[:,[rand_c_1,rand_c_2]]=chromosome[:,[rand_c_2,rand_c_1]]
            
        if fixed_map is not None:
            chromosome[fixed_map!=0]=fixed_map[fixed_map!=0]     
            
        target=target_function(chromosome)        
        
        offspring[i]=chromosome
        target_updated[i]=target
    
    return offspring,np.array(target_updated) 

def mutation_MPO_MBO(offspring,target_updated,objects_idx, mutation_rate=0.1, mpo_mbo='mbo',target_function = target_function,fixed_map=None):
    for i in range (0, offspring.shape[0]):
        target=target_updated[i]
        chromosome=offspring[i]
        # print(chromosome)
        ngh_finder=nghxy_finder.GridNghFinder(0, 0, chromosome.shape[0]-1,chromosome.shape[1]-1)
        
        probability=int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
        if (probability < mutation_rate):
            gene=np.random.choice(objects_idx)
            # print(gene)
            rand_r=random.randint(0,chromosome.shape[0]-1)
            rand_c=random.randint(0,chromosome.shape[1]-1)            
            # print(rand_r,rand_c)
            nghs=ngh_finder.find(rand_r,rand_c)                                                
            ngh_vals=[chromosome[j[0],j[1]] for j in nghs]   
            i_val=chromosome[rand_r,rand_c]
            ngh_vals.remove(i_val)   

            # nghs_selection=[(i[0],i[1]) for i in nghs if (i[0],i[1]) not in [(rand_r-1,rand_c-1),(rand_r-1,rand_c+1)] ]
            nghs_lst=nghs.tolist()
            random.shuffle(nghs_lst)
            nghs_selection=[(i[0],i[1]) for i in nghs_lst[:-2]]
            nghs_selection.append((rand_r,rand_c))
            
            if mpo_mbo=='mpo':
                for idx in nghs_selection:
                    chromosome[idx]=gene
                    
            elif mpo_mbo=='mbo':
                while gene not in ngh_vals and i_val!=gene:
                    gene=np.random.choice(objects_idx)
                    rand_r=random.randint(0,chromosome.shape[0]-1)
                    rand_c=random.randint(0,chromosome.shape[1]-1)            

                    nghs=ngh_finder.find(rand_r,rand_c)                                                
                    ngh_vals=[chromosome[j[0],j[1]] for j in nghs]   
                    i_val=chromosome[rand_r,rand_c]
                    ngh_vals.remove(i_val)   
        
                # nghs_selection=[(i[0],i[1]) for i in nghs if (i[0],i[1]) not in [(rand_r-1,rand_c-1),(rand_r-1,rand_c+1)] ]    
                nghs_lst=nghs.tolist()
                random.shuffle(nghs_lst)
                nghs_selection=[(i[0],i[1]) for i in nghs_lst[:-2]]                   
                # print(nghs_selection)
                    
                for idx in nghs_selection:
                    chromosome[idx]=gene
                    
                    
            if fixed_map is not None:
                chromosome[fixed_map!=0]=fixed_map[fixed_map!=0]  
                
            target=target_function(chromosome)      
            
        target_updated[i]=target
        offspring[i]=chromosome
    
    return offspring,np.array(target_updated)
          
def genetic_algorithm_2d(objects_idx,rows_n=5,cols_n=5,population_size=5,elite=0, generations=50,mutation_rate=0.1,crossover_name='crossover_CBO',mutation_name='mutation_MPO_MBO',mpo_mbo='mbo',target_function= target_function,fixed_map=None,population_init=None, verbose=1):    
    random.seed(None)
    np.random.seed(None)    
    
    global count
    count=0
    
    # print(objects_idx)
    population,target=initial_population(objects_idx,population_size,rows_n,cols_n,target_function,fixed_map=fixed_map,population_init=population_init)
    fitness=fitness_function(target)  
    sorted_idx=target.argsort()
    elite_ind=np.copy(population[sorted_idx][0,:])    
    best_target=target[sorted_idx[0]]
    
    epoch={}
    while(count<generations):
        if verbose:
            if count%verbose==0:            
                print(f'Generation = { count};\t f(x) = {best_target}' )  
        epoch[count]=best_target      
        
        offspring,target_updated=breeding(population, target,fitness,crossover_name, elite, target_function,fixed_map=fixed_map)         
        # print(target_updated)
        if mutation_name=='mutation_tsai_2':
            population,target=mutation_tsai_2(offspring,target_updated, mutation_rate,target_function,fixed_map=fixed_map)
        elif mutation_name=='mutation_tsai_1':
            population,target=mutation_tsai_1(offspring,target_updated, mutation_rate,target_function,fixed_map=fixed_map)
        elif mutation_name=='mutation_MPO_MBO':
            population,target=mutation_MPO_MBO(offspring,target_updated, objects_idx,mutation_rate,mpo_mbo,target_function,fixed_map=fixed_map)

        fitness=fitness_function(target)
        sorted_idx_=target.argsort()
        value=np.copy(population[sorted_idx_][0,:])   
        best_target_=target[sorted_idx_[0]]
        if best_target>best_target_:
            elite_ind=np.copy(value)    
            best_target=best_target_
            
        count+=1    
        # break

    return elite_ind,epoch
    # return 0
  

if __name__=="__main__":
    ###########################################################################   
    import matplotlib
    
    nlcd_class_idNcolor={
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
    nlcd_class_color={v[0]:v[1] for v in nlcd_class_idNcolor.values()}
    cmap_LC, norm=matplotlib.colors.from_levels_and_colors(list(nlcd_class_color.keys()),list(nlcd_class_color.values()),extend='max')
    
    
    jisperveld_data=usda_datasets.load_jisperveld_data()
    jisperveld_lu=jisperveld_data['lu']
    jisperveld_lu_name=jisperveld_data['lu_name']
    print(jisperveld_lu_name)
    usda_vis.imshow_label2darray(jisperveld_lu,figsize=(7,7),cmap=cmap_LC,norm=norm,fontsize=10)    
    
    objects_idx=list(range(1,10))
    rows_n=20
    cols_n=20  
    fixed_map=jisperveld_data['fixed_LU']
    
    #cost_dict={'natrue_cost':nature_cost,'recreation_cost':recreation_cost,'lu_conversion_cost':conversion_cost,'C':C,'L':L,'R':R} 
    cost_filter=['natrue_cost','recreation_cost','lu_conversion_cost','C','L','R','lu_area_cost']
    # cost_filter=['R']
 
    pattern_generated,epoch=genetic_algorithm_2d(
        objects_idx,
        rows_n=rows_n,
        cols_n=cols_n,
        population_size=50,
        generations=50,
        mutation_rate=0.5,
        target_function=target_function,
        crossover_name='crossover_CBO',
        mutation_name='mutation_MPO_MBO', 
        mpo_mbo='mpo',
        fixed_map=fixed_map,
        population_init=jisperveld_lu,
        verbose=10)

    usda_vis.imshow_label2darray(pattern_generated,figsize=(7,7),cmap=cmap_LC,norm=norm,fontsize=10) 
