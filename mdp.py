# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import rv_discrete

"""
Functions for the definition of the environment are described here
"""

############ DEFINE MDP ########### 
def sampleMatrices(n_states,n_actions,n_features) : 
    """
    The matrices (transition probabilities, rewards, features) are generated only once
    """
    
    # sample policies
    mu = np.random.rand(n_states,n_actions)
    mu = mu / np.sum(mu,axis = 1)[:,None]
    pi = np.random.rand(n_states,n_actions)
    pi = pi / np.sum(pi,axis = 1)[:,None]
    
    # sample transitions: sum over all the possible next states pf p(s'|s,a) = 1
    P_3D = np.random.rand(n_states,n_actions,n_states) + 0.00001
    P_3D = P_3D / np.sum(P_3D,axis = 2)[:,:,None]
    
    # sample rewards
    r = np.random.rand(n_states,n_actions)
   
    # sample features
    Phi = np.ones((n_states,n_features))
    Phi[:,:-1] = np.random.rand(n_states,n_features-1)
      
    return mu,pi,P_3D,r,Phi

def getTransition(P_3D,policy):
    # policy has to be a policy matrix |S| x |A|
    P = np.sum(P_3D * policy[:,:,None],axis = 1)
    return P

def getRewardVector(r,policy):
    # policy has to be a policy matrix |S| x |A|
    R = np.sum(r * policy,axis = 1)
    R = R[:,None]
    return R

def getRMSPBE(R,P,Phi,theta,projectionMatrix,D,discount) :
    v = Phi.dot(theta)
    tv = R + discount * P.dot(Phi.dot(theta))
    pitv = projectionMatrix.dot(tv)
    #measure = 0
    #for i in range(len(d)) : 
    #    measure += d[i]*(v[i] - pitv[i])**2
    measure = ((v - pitv).T).dot(D).dot(v-pitv)
    return float(measure)

def getAction(s,n_actions,policy) : 
    # policy has to be a policy matrix |S| x |A|
    d_a = list(policy[s])
    x = range(1,n_actions+1)
    a = int(rv_discrete(values=(x,d_a)).rvs(size=1))
    return a - 1

def getNewState(s,a,transitionProba,n_states) : 
    # transition Proba has to be a policy matrix |S| x |A| x |S|. The action is determined by the policy
    d_new_s = list(transitionProba[s,a])
    x = range(1,n_states+1)
    new_s = int(rv_discrete(values=(x,d_new_s)).rvs(size=1))
    return new_s - 1

# function that estimates the infinite distribution of states
def getDistribution(policy,transitionProba,n_actions,n_states,steps):
    # policy has to be a policy matrix |S| x |A|
    # transition Proba has to be a policy matrix |S| x |A| x |S|
    s = np.random.randint(0,n_states)
    count_step = 0
    distribution_states = [0]*n_states
    distribution_states[s] += 1
    while count_step < steps : 
        # take action and observe new state
        a = getAction(s,n_actions,policy)
        new_s = getNewState(s,a,transitionProba,n_states)
        distribution_states[new_s] += 1

        # go to new state
        s = new_s
        count_step += 1
    distribution_states = [elt/float(steps) for elt in distribution_states]
    return distribution_states


