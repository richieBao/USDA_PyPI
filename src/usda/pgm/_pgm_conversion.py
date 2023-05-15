# -*- coding: utf-8 -*-
"""
Created on Thu May 11 09:06:40 2023

@author: richie bao
"""
from pgmpy.models import BayesianNetwork

def convert_pgm_to_pgmpy(pgm):
    """Takes a Daft PGM object and converts it to a pgmpy BayesianModel"""
    edges = [(edge.node1.name, edge.node2.name) for edge in pgm._edges]
    model = BayesianNetwork(edges)
    return model