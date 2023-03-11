# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 19:58:43 2023

@author: richie bao

ref: Segaran, T. (2007). Programming Collective Intelligence (First). O’Reilly.
"""
import random
import math

# Mutation Operation
def mutate(vec,step,domain):
    i=random.randint(0,len(domain)-1)
    if random.random()<0.5 and vec[i]>domain[i][0]:
        return vec[0:i]+[vec[i]-step]+vec[i+1:] 
    elif vec[i]<domain[i][1]:
        return vec[0:i]+[vec[i]+step]+vec[i+1:]
    else:
        return vec

# Crossover Operation
def crossover(r1,r2,domain):
    i=random.randint(1,len(domain)-2)
    return r1[0:i]+r2[i:]

def genetic_algorithm_SegarantT(domain,costf,popsize=50,step=1,mutprob=0.2,elite=0.2,maxiter=100,verbose=1):
    # Build the initial population
    pop=[]
    for i in range(popsize):
        vec=[random.randint(domain[i][0],domain[i][1]) for i in range(len(domain))]
        pop.append(vec)
    
    # How many winners from each generation?
    topelite=int(elite*popsize)
    # print(pop)
    # Main loop 
    epoch={}
    for i in range(maxiter):
        # print(pop)
        # print('---',i)
        scores=[(costf(v),v) for v in pop]
        
        scores.sort()
        ranked=[v for (s,v) in scores]
        
        # Start with the pure winners
        pop=ranked[0:topelite]
        
        # Add mutated and bred forms of the winners
        while len(pop)<popsize:
            if random.random()<mutprob:        
                # Mutation
                c=random.randint(0,topelite)
                pop.append(mutate(ranked[c],step,domain))
            else:            
                # Crossover
                c1=random.randint(0,topelite)
                c2=random.randint(0,topelite)
                pop.append(crossover(ranked[c1],ranked[c2],domain))
        # print(pop)
        # Print current best score
        
        if verbose:
            if i%verbose==0:            
                print(f'iter_{i}: cost={scores[0][0]}')             
       
        epoch[i]=scores[0][0]
      
    return scores[0][1],epoch    
    

if __name__=="__main__":
    # The dorms, each of which has two available spaces
    dorms=['Zeus','Athena','Hercules','Bacchus','Pluto']
    
    # People, along with their first and second choices
    prefs=[('Toby', ('Bacchus', 'Hercules')),
           ('Steve', ('Zeus', 'Pluto')),
           ('Karen', ('Athena', 'Zeus')),
           ('Sarah', ('Zeus', 'Pluto')),
           ('Dave', ('Athena', 'Bacchus')), 
           ('Jeff', ('Hercules', 'Pluto')), 
           ('Fred', ('Pluto', 'Athena')), 
           ('Suzie', ('Bacchus', 'Hercules')), 
           ('Laura', ('Bacchus', 'Hercules')), 
           ('James', ('Hercules', 'Athena'))]
    
    # [(0,9),(0,8),(0,7),(0,6),...,(0,0)]
    domain=[(0,(len(dorms)*2)-i-1) for i in range(0,len(dorms)*2)]
    
    def printsolution(vec):
      slots=[]
      # Create two slots for each dorm
      for i in range(len(dorms)): slots+=[i,i]
      # print(slots)
      # Loop over each students assignment
      for i in range(len(vec)):
        x=int(vec[i])
    
        # Choose the slot from the remaining ones
        dorm=dorms[slots[x]]
        # Show the student and assigned dorm
        print (prefs[i][0],dorm)
        # Remove this slot
        del slots[x]
    
    def dormcost(vec):
      # print(vec)
      cost=0
      # Create list a of slots
      slots=[0,0,1,1,2,2,3,3,4,4]
      # print('+++',vec)
      # Loop over each student
      for i in range(len(vec)):
        # print('+++',i) 
        # print(vec[i])
        x=int(vec[i])
        dorm=dorms[slots[x]]
        pref=prefs[i][1]
        # First choice costs 0, second choice costs 1
        if pref[0]==dorm: cost+=0
        elif pref[1]==dorm: cost+=1
        else: cost+=3
        # Not on the list costs 3
    
        # Remove selected slot
        del slots[x]
        
      return cost
    
    best_score,epoch=genetic_algorithm_SegarantT(domain,dormcost,maxiter=100,verbose=10)
    # print(best_score)
    printsolution(best_score)
    