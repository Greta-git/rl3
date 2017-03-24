################## DEFINE ALGORITHMS ####################

import numpy as np

"""
Definition of updates for the algorithms
"""

def tdError(r,Phi,s,new_s,theta,discount) : 
    return r + discount * theta.T.dot(Phi[new_s]) - theta.T.dot(Phi[new_s])

def tdcUpdate(w0,theta0,tdErr,Phi,new_s,s,alpha,beta,discount,rho,offp) :
    if offp == False: 
        theta = theta0 + alpha * float(tdErr) * Phi[s][:,None] - alpha * discount * Phi[new_s][:,None] * (Phi[s][:,None].T.dot(w0))
        w = w0 + beta * (tdErr - Phi[s][:,None].T.dot(w0))*Phi[s][:,None]
    if offp == True:
        theta = theta0 + alpha * rho * float(tdErr) * Phi[s][:,None] - rho * alpha * discount * Phi[new_s][:,None] * (Phi[s][:,None].T.dot(w0))
        w = w0 + beta * (rho * tdErr - Phi[s][:,None].T.dot(w0))*Phi[s][:,None]        
    return theta,w

def tdUpdate(theta0,tdErr,Phi,s,alpha,rho,offp) : 
    if offp == False: 
        theta = theta0 + alpha * float(tdErr) * Phi[s][:,None]
    if offp == True: 
        theta = theta0 + alpha * rho * float(tdErr) * Phi[s][:,None]
    return theta
