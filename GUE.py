#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:57:46 2024

@author: ashtonlowenstein
"""

import numpy as np
from numpy.random import rand
from numpy.linalg import matrix_power
from numpy.linalg import eigvals
import matplotlib.pyplot as plt

#%% Functions

def initialstate(n):
    #Make a random starting Hermitian matrix where each element is between [-1,1)
    part = 2.*rand(n,n) - np.ones((n,n)) + 2.j*rand(n,n) - 1.j*np.ones((n,n))
    partbar = np.conj(part)
    state = (part + np.transpose(partbar))/2
    return state

def potential(M):
    #The interaction potential of the matrix model, acts as the energy in statistical mechanics
    arg = matrix_power(M,2)
    V = np.trace(arg)
    return -V
    
def mcmove(M):
    #Monte Carlo step using the Metropolis algorithm
    N = len(M)
    V = potential(M)
    delta = 0.075 #step size if any element is changed during MC
    for i in range(N):
        for k in range(N):
            a = np.random.randint(0,N)
            b = np.random.randint(0,N)
            z = delta*(2.*rand() - 1.0 + 2.j*rand() - 1.j)
            test = np.copy(M)
            test[a,b] = test[a,b] + z
            test[b,a] = test[b,a] + np.conj(z)
            if potential(test) < V:
                M[a,b] = test[a,b]
                M[b,a] = test[b,a]
                return M
            elif rand() < np.exp(N*potential(M)):
                M[a,b] = test[a,b]
                M[b,a] = test[b,a]
                return M
            else:
                return M


#%% Testing

#A = initialstate(2)
#print(A)
#print(mcmove(A))

#print(np.real(eigvals(A)))

#%% Generating GUE sample

N_size = 100 #size of matrix
N_eq = 10000 #number of steps to eqlb
N_sample = 20000 #number of samples in constructed ensemble
#L = N_size*N_size*N_sample

Evals = np.zeros((N_sample,N_size))

M = initialstate(N_size)

for s in range(N_eq):
    mcmove(M)
    
for t in range(N_sample):
    mcmove(M)
    Evals[t] = np.real(eigvals(M))

Evals = np.ndarray.flatten(Evals)

#%% Plots

fig, ax = plt.subplots()
ax.hist(Evals, bins=55, linewidth=.1)
ax.set(xlim=(-15,15))
ax.text(-12,50000,r'$\delta = 0.075$')
plt.show()