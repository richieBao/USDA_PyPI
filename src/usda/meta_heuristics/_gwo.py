# -*- coding: utf-8 -*-
############################################################################
# 引自：pyDecision库：https://github.com/Valdecy/pyDecision
# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Grey Wolf Optimizer

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
def initial_position(pack_size = 5, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    position = np.zeros((pack_size, len(min_values)+1))
    for i in range(0, pack_size):
        for j in range(0, len(min_values)):
             position[i,j] = random.uniform(min_values[j], max_values[j])
        position[i,-1] = target_function(position[i,0:position.shape[1]-1])
    return position

# Function: Initialize Alpha
def alpha_position(dimension = 2, target_function = target_function):
    alpha = np.zeros((1, dimension + 1))
    for j in range(0, dimension):
        alpha[0,j] = 0.0
    alpha[0,-1] = target_function(alpha[0,0:alpha.shape[1]-1])
    return alpha

# Function: Initialize Beta
def beta_position(dimension = 2, target_function = target_function):
    beta = np.zeros((1, dimension + 1))
    for j in range(0, dimension):
        beta[0,j] = 0.0
    beta[0,-1] = target_function(beta[0,0:beta.shape[1]-1])
    return beta

# Function: Initialize Delta
def delta_position(dimension = 2, target_function = target_function):
    delta =  np.zeros((1, dimension + 1))
    for j in range(0, dimension):
        delta[0,j] = 0.0
    delta[0,-1] = target_function(delta[0,0:delta.shape[1]-1])
    return delta

# Function: Updtade Pack by Fitness
def update_pack(position, alpha, beta, delta):
    updated_position = np.copy(position)
    for i in range(0, position.shape[0]):
        if (updated_position[i,-1] < alpha[0,-1]):
            alpha[0,:] = np.copy(updated_position[i,:])
        if (updated_position[i,-1] > alpha[0,-1] and updated_position[i,-1] < beta[0,-1]):
            beta[0,:] = np.copy(updated_position[i,:])
        if (updated_position[i,-1] > alpha[0,-1] and updated_position[i,-1] > beta[0,-1]  and updated_position[i,-1] < delta[0,-1]):
            delta[0,:] = np.copy(updated_position[i,:])
    return alpha, beta, delta

# Function: Updtade Position
def update_position(position, alpha, beta, delta, a_linear_component = 2, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    updated_position = np.copy(position)
    for i in range(0, updated_position.shape[0]):
        for j in range (0, len(min_values)):   
            r1_alpha              = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            r2_alpha              = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            a_alpha               = 2*a_linear_component*r1_alpha - a_linear_component
            c_alpha               = 2*r2_alpha      
            distance_alpha        = abs(c_alpha*alpha[0,j] - position[i,j]) 
            x1                    = alpha[0,j] - a_alpha*distance_alpha   
            r1_beta               = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            r2_beta               = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            a_beta                = 2*a_linear_component*r1_beta - a_linear_component
            c_beta                = 2*r2_beta            
            distance_beta         = abs(c_beta*beta[0,j] - position[i,j]) 
            x2                    = beta[0,j] - a_beta*distance_beta                          
            r1_delta              = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            r2_delta              = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            a_delta               = 2*a_linear_component*r1_delta - a_linear_component
            c_delta               = 2*r2_delta            
            distance_delta        = abs(c_delta*delta[0,j] - position[i,j]) 
            x3                    = delta[0,j] - a_delta*distance_delta                                 
            updated_position[i,j] = np.clip(((x1 + x2 + x3)/3),min_values[j],max_values[j])     
        updated_position[i,-1] = target_function(updated_position[i,0:updated_position.shape[1]-1])
    return updated_position

############################################################################

# GWO Function
def grey_wolf_optimizer(pack_size = 5, min_values = [-5,-5], max_values = [5,5], iterations = 50, target_function = target_function, verbose = 1):    
    count    = 0
    alpha    = alpha_position(dimension = len(min_values), target_function = target_function)
    beta     = beta_position(dimension  = len(min_values), target_function = target_function)
    delta    = delta_position(dimension = len(min_values), target_function = target_function)
    position = initial_position(pack_size = pack_size, min_values = min_values, max_values = max_values, target_function = target_function)
    while (count <= iterations): 
        if verbose:
            if count%verbose==0:
                print('Iteration = ', count, ' f(x) = ', alpha[0][-1])      
        a_linear_component = 2 - count*(2/iterations)
        alpha, beta, delta = update_pack(position, alpha, beta, delta)
        position           = update_position(position, alpha, beta, delta, a_linear_component = a_linear_component, min_values = min_values, max_values = max_values, target_function = target_function)    
        count              = count + 1          
    return alpha

############################################################################
# Function: BWM
def bw_method(mic, lic, size = 50, iterations = 150,verbose=1):
    best      = np.where(mic == 1)[0][0]
    worst     = np.where(lic == 1)[0][0]
    pairs_b   = [(best, i)  for i in range(0, mic.shape[0])]
    pairs_w   = [(i, worst) for i in range(0, mic.shape[0]) if (i, worst) not in pairs_b]
    def target_function(variables):
        eps       = [float('+inf')]
        for pair in pairs_b:
            i, j = pair
            diff = abs(variables[i] - variables[j]*mic[j])
            if ( i != j):
                eps.append(diff)
        for pair in pairs_w:
            i, j = pair
            diff = abs(variables[i] - variables[j]*lic[j])
            if ( i != j):
                eps.append(diff)
        if ( np.sum(variables) == 0):
            eps = float('+inf')
        else:
            eps = max(eps[1:])
        return eps
    weights = grey_wolf_optimizer(pack_size = size, min_values = [0.01]*len(mic), max_values = [1]*len(mic), iterations = iterations, target_function = target_function,verbose=verbose)
    weights = weights[0][:-1]/sum(weights[0][:-1])
    return weights
if __name__=="__main__":
    # BWM     
    # Dataset
    dataset = np.array([
                    # g1   g2   g3   g4
                    [250,  16,  12,  5],
                    [200,  16,  8 ,  3],   
                    [300,  32,  16,  4],
                    [275,  32,  8 ,  4],
                    [225,  16,  16,  2]
                    ])
    
    # Most Important Criteria
    mic = np.array([1, 3, 4, 7])
    
    # Least Important Criteria
    lic = np.array([7, 5, 5, 1])
    
    weights =  bw_method( mic, lic, size = 50, iterations = 150,verbose=50)
    print(weights)