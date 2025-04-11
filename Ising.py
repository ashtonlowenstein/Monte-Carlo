#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 10:47:02 2024

@author: ashtonlowenstein
"""

import numpy as np
from numpy.random import rand
import random
import matplotlib.pyplot as plt

#%% Functions

#All plus state
#Plus_state = np.ones((N_spin,N_spin))

#All minus state
#Minus_state = -np.copy(Plus_state)

#Randomly chosen initial global (micro)state
def make_initial_state(n):
    state = 2*np.random.randint(2, size=(n,n))-1
    return state

#nearest neighbor sum
def nn_sum(A,i,j):
    n = len(A)
    return A[i,(j+1)%n] + A[(i+1)%n,j] + A[(i-1)%n,j] + A[i,(j-1)%n]


#Hamiltonian for Ising model
def Hamiltonian(A):
    H = 0
    n = len(A)
    for i in range(n):
        for j in range(n):
            H += -A[i,j]*nn_sum(A,i,j)
    return H/4

#Magnetization
def mag(A):
    m = np.sum(A)
    return m

#Possible energy changes after flipping one spin based on nearest neighbor interaction
#delta_E = {4:16, 3:8, 2:0, 1:-8, 0:-16}


#Number of nearest neighbor spins in common
def nn_in_common(A,i,j):
    common = 0
    s = [-1,1]
    n = len(A)
    
    for p in s:
        if A[i,j] == A[i,(j+p)%n]:
            common += 1
        else:
            pass
        
    for q in s:
        if A[i,j] == A[(i+q)%n,j]:
            common += 1
        else:
            pass
    
    return common

#Single monte carlo step using the Metropolis algorithm
def MC_move(A,beta):
    delta_E = {4:8, 3:4, 2:0, 1:-4, 0:-8}
    i1 = random.choice(range(len(A)))
    j1 = random.choice(range(len(A)))
    config = np.copy(A) 
    common = nn_in_common(config,i1,j1)
    deltaE = config[i1,j1]*delta_E[common]
    if deltaE < 0:
        config[i1,j1] = -config[i1,j1]
    elif rand() < np.exp(-beta*deltaE):
        config[i1,j1] = -config[i1,j1]
    else:
        pass
    return config

def mcmove(config, beta):
    '''Monte Carlo move using Metropolis algorithm '''
    N = len(config)
    for i in range(N):
        for j in range(N):
                a = np.random.randint(0, N)
                b = np.random.randint(0, N)
                s =  config[a, b]
                #nb = config[(a+1)%N,b] + config[a,(b+1)%N] + config[(a-1)%N,b] + config[a,(b-1)%N]
                nb = nn_sum(config,a,b)
                cost = 2*s*nb
                if cost < 0:
                    s *= -1
                elif random.random() < np.exp(-cost*beta):
                    s *= -1
                config[a, b] = s
    return config

def calcEnergy(config):
    '''Energy of a given configuration'''
    N = len(config)
    energy = 0
    for i in range(len(config)):
        for j in range(len(config)):
            S = config[i,j]
            nb = config[(i+1)%N, j] + config[i,(j+1)%N] + config[(i-1)%N, j] + config[i,(j-1)%N]
            energy += -nb*S
    return energy/4

#%% Ising Simulation

#J = 1 #Coupling constant
#b = 0 #External magnetic field
#beta = .25 #1/kB*T

#kB = 1.38e-23

N_spin = 20 #Total number of spins
N_eq = 2000 #Number of initial steps to equilibrate
N_steps = 1024 #Number of steps for sampling
n1 = 1.0/(N_steps*N_spin*N_spin)
n2 = 1.0/(N_steps*N_steps*N_spin*N_spin)
N_pts = 24

T = np.linspace(1.53, 3.28, N_pts)
C = np.zeros(N_pts)


#for i in range(N_pts):
#    E1 = 0
#    E2 = 0
#    config = make_initial_state(N_spin)
#    b = 1/T[i]
    
#    for j in range(N_steps):
#        mcmove(config,b)
        
#    for k in range(N_steps):
#        mcmove(config,b)
#        Energy = calcEnergy(config)
#        E1 = E1 + Energy
#        E2 = E2 + Energy*Energy 
        
#    C[i]=(n1*E2 - n2*E1*E1)/(b**2)
    
for tt in range(N_pts):
    E1 = E2 = 0
    config = make_initial_state(N_spin)
    iT=1.0/T[tt]; iT2=iT*iT;
    
    for i in range(N_steps):         # equilibrate
        mcmove(config, iT)           # Monte Carlo moves

    for i in range(N_steps):
        mcmove(config, iT)           
        Ene = calcEnergy(config)     # calculate the energy

        E1 = E1 + Ene
        E2 = E2 + Ene*Ene

    C[tt] = (n1*E2 - n2*E1*E1)*iT2
    
    
#%% Plots

fig, ax = plt.subplots()
ax.scatter(T, C, marker='.')
ax.set_title('Specific Heat vs. Temperature')
ax.set_xlabel(r'$T$')
ax.set_ylabel(r'$c$')
ax.text(2.8,0.35,r'$N = 20$')
plt.show()
    
#fig, ax = plt.subplots()
#ax.imshow(State_History, origin='lower')
#plt.show()