############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Genetic Algorithm

# PEREIRA, V. (2022). GitHub repository: https://github.com/Valdecy/pyMetaheuristic

############################################################################

# Required Libraries
import numpy  as np
import math
import random
import os

############################################################################

# Function
def target_function():
    return

############################################################################

# Function: Initialize Variables
def initial_population(population_size = 5, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    population = np.zeros((population_size, len(min_values) + 1))
    for i in range(0, population_size):
        for j in range(0, len(min_values)):
             population[i,j] = random.uniform(min_values[j], max_values[j]) 
        population[i,-1] = target_function(population[i,0:population.shape[1]-1])
    return population

# Function: Fitness
def fitness_function(population): 
    fitness = np.zeros((population.shape[0], 2))
    for i in range(0, fitness.shape[0]):
        fitness[i,0] = 1/(1+ population[i,-1] + abs(population[:,-1].min()))
    fit_sum = fitness[:,0].sum()
    fitness[0,1] = fitness[0,0]
    for i in range(1, fitness.shape[0]):
        fitness[i,1] = (fitness[i,0] + fitness[i-1,1])
    for i in range(0, fitness.shape[0]):
        fitness[i,1] = fitness[i,1]/fit_sum
    return fitness

# Function: Selection
def roulette_wheel(fitness): 
    ix = 0
    random = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
    for i in range(0, fitness.shape[0]):
        if (random <= fitness[i, 1]):
          ix = i
          break
    return ix

# Function: Offspring
def breeding(population, fitness, min_values = [-5,-5], max_values = [5,5], mu = 1, elite = 0, target_function = target_function):
    offspring = np.copy(population)
    b_offspring = 0
    if (elite > 0):
        preserve = np.copy(population[population[:,-1].argsort()])
        for i in range(0, elite):
            for j in range(0, offspring.shape[1]):
                offspring[i,j] = preserve[i,j]
    for i in range (elite, offspring.shape[0]):
        parent_1, parent_2 = roulette_wheel(fitness), roulette_wheel(fitness)
        while parent_1 == parent_2:
            parent_2 = random.sample(range(0, len(population) - 1), 1)[0]
        for j in range(0, offspring.shape[1] - 1):
            rand = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            rand_b = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)                                
            if (rand <= 0.5):
                b_offspring = 2*(rand_b)
                b_offspring = b_offspring**(1/(mu + 1))
            elif (rand > 0.5):  
                b_offspring = 1/(2*(1 - rand_b))
                b_offspring = b_offspring**(1/(mu + 1))       
            offspring[i,j] = np.clip(((1 + b_offspring)*population[parent_1, j] + (1 - b_offspring)*population[parent_2, j])/2, min_values[j], max_values[j])           
            if(i < population.shape[0] - 1):   
                offspring[i+1,j] = np.clip(((1 - b_offspring)*population[parent_1, j] + (1 + b_offspring)*population[parent_2, j])/2, min_values[j], max_values[j]) 
        offspring[i,-1] = target_function(offspring[i,0:offspring.shape[1]-1]) 
    return offspring
 
# Function: Mutation
def mutation(offspring, mutation_rate = 0.1, eta = 1, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    d_mutation = 0            
    for i in range (0, offspring.shape[0]):
        for j in range(0, offspring.shape[1] - 1):
            probability = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            if (probability < mutation_rate):
                rand = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
                rand_d = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)                                     
                if (rand <= 0.5):
                    d_mutation = 2*(rand_d)
                    d_mutation = d_mutation**(1/(eta + 1)) - 1
                elif (rand > 0.5):  
                    d_mutation = 2*(1 - rand_d)
                    d_mutation = 1 - d_mutation**(1/(eta + 1))                
                offspring[i,j] = np.clip((offspring[i,j] + d_mutation), min_values[j], max_values[j])
        offspring[i,-1] = target_function(offspring[i,0:offspring.shape[1]-1])                        
    return offspring

############################################################################

# GA Function
def genetic_algorithm(population_size = 5, mutation_rate = 0.1, elite = 0, min_values = [-5,-5], max_values = [5,5], eta = 1, mu = 1, generations = 50, target_function = target_function, verbose = 1):    
    count = 0
    population = initial_population(population_size = population_size, min_values = min_values, max_values = max_values, target_function = target_function)
    fitness = fitness_function(population)    
    elite_ind = np.copy(population[population[:,-1].argsort()][0,:])
    
    epoch={}
    while (count <= generations):  
        if verbose:
            if count%verbose==0:            
                print('Generation = ', count, ' f(x) = ', elite_ind[-1])  
        epoch[count]=elite_ind[-1]  
        
        offspring = breeding(population, fitness, min_values = min_values, max_values = max_values, mu = mu, elite = elite, target_function = target_function) 
        population = mutation(offspring, mutation_rate = mutation_rate, eta = eta, min_values = min_values, max_values = max_values, target_function = target_function)
        fitness = fitness_function(population)
        value = np.copy(population[population[:,-1].argsort()][0,:])
        if(elite_ind[-1] > value[-1]):
            elite_ind = np.copy(value) 
        count = count + 1       
    return elite_ind, epoch

############################################################################

# Required Libraries
import matplotlib.pyplot as plt


# Function: Rank 
def ranking(flow):    
    rank_xy = np.zeros((flow.shape[0], 2))
    for i in range(0, rank_xy.shape[0]):
        rank_xy[i, 0] = 0
        rank_xy[i, 1] = flow.shape[0]-i           
    for i in range(0, rank_xy.shape[0]):
        plt.text(rank_xy[i, 0],  rank_xy[i, 1], 'a' + str(int(flow[i,0])), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0.8, 1.0, 0.8),))
    for i in range(0, rank_xy.shape[0]-1):
        plt.arrow(rank_xy[i, 0], rank_xy[i, 1], rank_xy[i+1, 0] - rank_xy[i, 0], rank_xy[i+1, 1] - rank_xy[i, 1], head_width = 0.01, head_length = 0.2, overhang = 0.0, color = 'black', linewidth = 0.9, length_includes_head = True)
    axes = plt.gca()
    axes.set_xlim([-1, +1])
    ymin = np.amin(rank_xy[:,1])
    ymax = np.amax(rank_xy[:,1])
    if (ymin < ymax):
        axes.set_ylim([ymin, ymax])
    else:
        axes.set_ylim([ymin-1, ymax+1])
    plt.axis('off')
    plt.show() 
    return

# Function: IDOCRIW
def idocriw_method(dataset, criterion_type, size = 20, gen = 12000, graph = True,verbose = 1):
    X    = np.copy(dataset)
    X    = X/X.sum(axis = 0)
    X_ln = np.copy(dataset)
    X_r  = np.copy(dataset)
    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[1]):
            X_ln[i,j] = X[i,j]*math.log(X[i,j])
    d    = np.zeros((1, X.shape[1]))
    w    = np.zeros((1, X.shape[1]))
    for i in range(0, d.shape[1]):
        d[0,i] = 1-( -1/(math.log(d.shape[1]))*sum(X_ln[:,i])) 
    for i in range(0, w.shape[1]):
        w[0,i] = d[0,i]/d.sum(axis = 1)
    for i in range(0, len(criterion_type)):
        if (criterion_type[i] == 'min'):
           X_r[:,i] = dataset[:,i].min() / X_r[:,i]
    X_r   = X_r/X_r.sum(axis = 0)
    #a_min = X_r.min(axis = 0)       
    a_max = X_r.max(axis = 0) 
    A     = np.zeros(dataset.shape)
    np.fill_diagonal(A, a_max)

    for k in range(0, A.shape[0]):
        i, _ = np.where(X_r == a_max[k])
        i    = i[0]
        for j in range(0, A.shape[1]):
            A[k, j] = X_r[i, j]
    #a_min_ = A.min(axis = 0)       
    a_max_ = A.max(axis = 0) 
    P      = np.copy(A)    
    for i in range(0, P.shape[1]):
        P[:,i] = (-P[:,i] + a_max_[i])/a_max[i]
    WP     = np.copy(P)
    np.fill_diagonal(WP, -P.sum(axis = 0))
    
    ################################################
    def target_function(variable = [0]*WP.shape[1]):
        variable = [variable[i]/sum(variable) for i in range(0, len(variable))]
        WP_s     = np.copy(WP)
        for i in range(0, WP.shape[0]):
            for j in range(0, WP.shape[1]):
                WP_s[i, j] = WP_s[i, j]*variable[j]
        total = abs(WP_s.sum(axis = 1)) 
        total = sum(total) 
        return total
    ################################################
    
    solution = genetic_algorithm(population_size = size, mutation_rate = 0.1, elite = 1, min_values = [0]*WP.shape[1], max_values = [1]*WP.shape[1], eta = 1, mu = 1, generations = gen, target_function = target_function,verbose = verbose)
    solution = solution[:-1]
    solution = solution/sum(solution)
    print(f'solution:{solution}')
    w_       = np.copy(w)
    w_       = w_*solution
    w_       = w_/w_.sum()
    w_       = w_.T
    for i in range(0, w_.shape[0]):
        print('a' + str(i+1) + ': ' + str(round(w_[i][0], 4)))
    if ( graph == True):
        flow = np.copy(w_)
        flow = np.reshape(flow, (w_.shape[0], 1))
        flow = np.insert(flow, 0, list(range(1, w_.shape[0]+1)), axis = 1)
        flow = flow[np.argsort(flow[:, 1])]
        flow = flow[::-1]
        ranking(flow)
    return w_

if __name__=="__main__":
    import array_to_latex as a2l
    # IDOCRIW
    
    # Criterion Type: 'max' or 'min'
    # criterion_type = ['max', 'max', 'max', 'min', 'min', 'min', 'min']
    
    # Dataset
    # dataset = np.array([
    #                     [75.5, 420,	 74.2, 2.8,	 21.4,	0.37,	 0.16],   #a1
    #                     [95,   91,	 70,	 2.68, 22.1,	0.33,	 0.16],   #a2
    #                     [770,  1365, 189,	 7.9,	 16.9,	0.04,	 0.08],   #a3
    #                     [187,  1120, 210,	 7.9,	 14.4,	0.03,	 0.08],   #a4
    #                     [179,  875,	 112,	 4.43,	9.4,	0.016, 0.09],   #a5
    #                     [239,  1190, 217,	 8.51,	11.5,	0.31,	 0.07],   #a6
    #                     [273,  200,	 112,	 8.53,	19.9,	0.29,	 0.06]    #a7
    #                     ])
    criterion_type=['min','max','min','max']
    dataset=np.array([[3,100,10,7],
                      [2.5,80,8,5],
                      [1.8,50,20,11],
                      #[2.2,70,12,9],
                      #[2.2,70,12,9],
                      [2.2,70,12,9]])
    
    
    rank = idocriw_method(dataset, criterion_type, size = 20, gen = 100, graph = True,verbose=10)
