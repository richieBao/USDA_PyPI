# -*- coding: utf-8 -*-
"""
Created on Tue May  2 21:02:43 2023

@author: richie bao
"""
import math
import numpy as np

def P_Xk_binomialDistribution(n,k,p):
    nCr=math.comb(n,k)
    P_Xk=nCr*p**k*(1-p)**(n-k)
    return P_Xk

def covariance_test(cov_matrix, col_matrix_range):
    
    a = np.random.choice(col_matrix_range, cov_matrix.shape[-1])
    eigvalues, eigvectors = np.linalg.eig(cov_matrix)
    
    results = {
        "Positive Definite": False,
        "Positive Semi-Definite": False,
        "Symmetric": False,
        "Positive Determinant": False,
        "Eigen Values, Positivity": False
    }
    
    if(a.T @ cov_matrix @ a > 0):
        results["Positive Definite"] = True
    if((a.T @ cov_matrix @ a == 0) or (a.T @ cov_matrix @ a > 0)):
        results["Positive Semi-Definite"] = True
    if(np.all(cov_matrix == cov_matrix.T)):
        results["Symmetric"] = True
    if(np.linalg.det(cov_matrix) >= 0):
        results["Positive Determinant"] = True
    if(-1 not in np.sign(eigvalues)):
        results["Eigen Values, Positivity"] = True
        
    return results

def print_results(result):
    for key, value in result.items():
        status = "Passed" if value else "Failed"
        print(f'{key}: {status}')