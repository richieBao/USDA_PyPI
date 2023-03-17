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
import numpy  as np
import random
import copy
import os
from ..pattern_signature import  _grid_neighbors_xy_finder as nghxy_finder

# Function
def target_function():
    return

def population_replace_fixed_map(population,fixed_map):
    pop_lst=[]
    for chromosome in np.copy(population):
        chromosome[fixed_map!=0]=fixed_map[fixed_map!=0]
        pop_lst.append(chromosome)
        
    return np.array(pop_lst)

def initial_population(object_idx,population_size=5, rows_n=5, cols_n=5,target_function=target_function,fixed_map=None,population_init=None):
    if population_init is not None:
        population=np.array([population_init]*population_size)
    else:
        population=np.random.choice(object_idx,(population_size, rows_n,cols_n))
    
    if fixed_map is not None:        
        population=population_replace_fixed_map(population,fixed_map)        
        
    target=np.array(list(map(target_function,population)))

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

def crossover_tsai(p_1,p_2,target_function=target_function):
    rand_r=random.randint(0,p_1.shape[0])
    rand_c=random.randint(0,p_1.shape[1])
    rand_a= int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1) 
    rand_b= int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1) 
    
    offspring_individual_1=np.copy(p_1)
    offspring_individual_2=np.copy(p_2)
    offspring_individual_3=np.copy(p_1)
    offspring_individual_4=np.copy(p_2)   

    idx_along_row=[(i,j) for i in range(p_1.shape[0]) for j in range(p_1.shape[1])]
    idx_along_col=[(i,j) for j in range(p_1.shape[1]) for i in range(p_1.shape[0])]
    
    if rand_a>0.5:     
        idx=idx_along_row[:rand_r*p_1.shape[0]+rand_c]
        for i in idx:
            offspring_individual_1[i]=p_2[i]
            offspring_individual_2[i]=p_1[i]
    else:
        idx=idx_along_col[:rand_r*p_1.shape[0]+rand_c]
        for i in idx:
            offspring_individual_3[i]=p_2[i]
            offspring_individual_4[i]=p_1[i]      
    choice_idx=np.random.choice([0,1,2,3])
    offspring_individuals=[offspring_individual_1,offspring_individual_2,offspring_individual_3,offspring_individual_4]
    offspring_individual=offspring_individuals[choice_idx]
    target=target_function(offspring_individual)        
    
    return offspring_individual,target

def crossover_CBO(p_1,p_2,target_function=target_function):
    rand_r_1=random.randint(0,p_1.shape[0]-1)
    rand_c_1=random.randint(0,p_1.shape[1]-1)    
    rand_r_2=random.randint(0,p_1.shape[0]-1)
    rand_c_2=random.randint(0,p_1.shape[1]-1) 
    
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
            offspring_individual,target=crossover_tsai(p_1,p_2,target_function)            
        elif crossover_name=='crossover_CBO':
            offspring_individual,target=crossover_CBO(p_1,p_2,target_function=target_function)        
        
        if fixed_map is not None:
            offspring_individual[fixed_map!=0]=fixed_map[fixed_map!=0]
            
        offspring[i]=offspring_individual
        target_updated.append(target)
    
    return offspring,target_updated

# Function: Mutation        
def mutation_tsai_1(offspring,target_updated, mutation_rate=0.1, target_function = target_function,fixed_map=None):
    for i in range (0, offspring.shape[0]):
        target=target_updated[i]
        chromosome=offspring[i]
        probability=int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)

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
        probability=int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
        
        rand_r_1=random.randint(0,chromosome.shape[0]-1)
        rand_c_1=random.randint(0,chromosome.shape[1]-1)    
        rand_r_2=random.randint(0,chromosome.shape[0]-1)
        rand_c_2=random.randint(0,chromosome.shape[1]-1)          
        
        while rand_r_1==rand_r_2:
            rand_r_2=random.randint(0,chromosome.shape[0]-1)
        while rand_c_1==rand_c_2:
            rand_c_2=random.randint(0,chromosome.shape[1]-1)   
        
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
        ngh_finder=nghxy_finder.GridNghFinder(0, 0, chromosome.shape[0]-1,chromosome.shape[1]-1)
        
        probability=int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
        if (probability < mutation_rate):
            gene=np.random.choice(objects_idx)
            rand_r=random.randint(0,chromosome.shape[0]-1)
            rand_c=random.randint(0,chromosome.shape[1]-1)            
            nghs=ngh_finder.find(rand_r,rand_c)                                                
            ngh_vals=[chromosome[j[0],j[1]] for j in nghs]   
            i_val=chromosome[rand_r,rand_c]
            ngh_vals.remove(i_val)   

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
        
                nghs_lst=nghs.tolist()
                random.shuffle(nghs_lst)
                nghs_selection=[(i[0],i[1]) for i in nghs_lst[:-2]]                   
                    
                for idx in nghs_selection:
                    chromosome[idx]=gene
                    
                    
            if fixed_map is not None:
                chromosome[fixed_map!=0]=fixed_map[fixed_map!=0]  
                
            target=target_function(chromosome)      
            
        target_updated[i]=target
        offspring[i]=chromosome
    
    return offspring,np.array(target_updated)
          
def genetic_algorithm_2d_fixed_map(objects_idx,rows_n=5,cols_n=5,population_size=5,elite=0, generations=50,mutation_rate=0.1,crossover_name='crossover_CBO',mutation_name='mutation_MPO_MBO',mpo_mbo='mbo',target_function= target_function,fixed_map=None,population_init=None, verbose=1):    
    random.seed(None)
    np.random.seed(None)    
    
    global count
    count=0

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

    return elite_ind,epoch
 

if __name__=="__main__": 
    objects_idx=list(range(1,10))
    rows_n=20
    cols_n=20  
    fixed_map=jisperveld_data['fixed_LU']
 
    pattern_generated,epoch=genetic_algorithm_2d_fixed_map(
        objects_idx,
        rows_n=rows_n,
        cols_n=cols_n,
        population_size=50,
        generations=100,
        mutation_rate=0.5,
        target_function=target_function,
        crossover_name='crossover_CBO',
        mutation_name='mutation_MPO_MBO', 
        mpo_mbo='mpo',
        fixed_map=fixed_map,
        population_init=jisperveld_lu,
        verbose=10)