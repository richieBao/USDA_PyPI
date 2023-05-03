# -*- coding: utf-8 -*-
"""
Created on Tue May  2 21:02:43 2023

@author: richie bao
"""
import math

def P_Xk_binomialDistribution(n,k,p):
    nCr=math.comb(n,k)
    P_Xk=nCr*p**k*(1-p)**(n-k)
    return P_Xk