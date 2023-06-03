# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:15:26 2023

@author: richie bao
ref: http://localhost:8889/notebooks/omen_richiebao/omen_ref/USDA/special%20exploration/MRF/pgmpy_notebook-master/notebooks/11.%20A%20Bayesian%20Network%20to%20model%20the%20influence%20of%20energy%20consumption%20on%20greenhouse%20gases%20in%20Italy.ipynb
"""
from pgmpy.independencies.Independencies import IndependenceAssertion
from pgmpy.inference import VariableElimination
import time

def active_trails_of(model,query, evidence):
    active = model.active_trail_nodes(query, observed=evidence).get(query)
    active.remove(query)
    if active:
        if evidence:
            print(f'Active trails between \'{query}\' and {active} given the evidence {set(evidence)}.')
        else:
            print(f'Active trails between \'{query}\' and {active} given no evidence.')
    else:
        print(f'No active trails for \'{query}\' given the evidence {set(evidence)}.')
        
def markov_blanket_of(model,node):
    print(f'Markov blanket of \'{node}\' is {set(model.get_markov_blanket(node))}')   

def check_assertion(model, independent, from_variables, evidence):
    assertion = IndependenceAssertion(independent, from_variables, evidence)
    result = False
    for a in model.get_independencies().get_assertions():
        if frozenset(assertion.event1) == a.event1 and assertion.event2 <= a.event2 and frozenset(assertion.event3) == a.event3:
            result = True
            break
    print(f'{assertion}: {result}')    
    
def query_report(infer, variables, evidence=None, elimination_order="MinFill", show_progress=False, desc=""):
    if desc:
        print(desc)
    start_time = time.time()
    print(infer.query(variables=variables, 
                      evidence=evidence, 
                      elimination_order=elimination_order, 
                      show_progress=show_progress))
    print(f'--- Query executed in {time.time() - start_time:0,.4f} seconds ---\n')
    
def get_ordering(infer, variables, evidence=None, elimination_order="MinFill", show_progress=False, desc=""):
    start_time = time.time()
    ordering = infer._get_elimination_order(variables=variables, 
                                        evidence=evidence, 
                                        elimination_order=elimination_order, 
                                        show_progress=show_progress)
    if desc:
        print(desc, ordering, sep='\n')
        print(f'--- Ordering found in {time.time() - start_time:0,.4f} seconds ---\n')
    return ordering

def padding(heuristic):
    return (heuristic + ":").ljust(16)

def compare_all_ordering(infer, variables, evidence=None, show_progress=False):
    ord_dict = {
        "MinFill": get_ordering(infer, variables, evidence, "MinFill", show_progress),
        "MinNeighbors": get_ordering(infer, variables, evidence, "MinNeighbors", show_progress),
        "MinWeight": get_ordering(infer, variables, evidence, "MinWeight", show_progress),
        "WeightedMinFill": get_ordering(infer, variables, evidence, "WeightedMinFill", show_progress)
    }
    if not evidence:
        pre = f'elimination order found for probability query of {variables} with no evidence:'
    else:
        pre = f'elimination order found for probability query of {variables} with evidence {evidence}:'
    if ord_dict["MinFill"] == ord_dict["MinNeighbors"] and ord_dict["MinFill"] == ord_dict["MinWeight"] and ord_dict["MinFill"] == ord_dict["WeightedMinFill"]:
        print(f'All heuristics find the same {pre}.\n{ord_dict["MinFill"]}\n')
    else:
        print(f'Different {pre}')
        for heuristic, order in ord_dict.items():
            print(f'{padding(heuristic)} {order}')
        print()    