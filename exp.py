############### EXPERIMENT ON RANDOM MDP ######### 

import numpy as np
import mdp
import algo

"""
This file contains basically the for loop to generate the experiments (number of steps per experience and many runs)
"""

def experimentMDP(maxSteps,n_states,n_features,n_actions,P_3D,pi,mu,discount,Phi,alpha,Ptarget,projMatrix,D,beta,offp,r_s_a):
    
    Rtarget = mdp.getRewardVector(r_s_a,pi)
    s = np.random.randint(0,n_states)
    count_step = 0

    thetaTD = np.zeros((n_features,1))
    thetaTDC = np.zeros((n_features,1))
    wTDC = np.zeros((n_features,1))

    perf_TDC = []
    perf_TD = []

    while count_step < maxSteps : 
        # take action, observe reward and go to new state
        if offp == True: 
            a = mdp.getAction(s,n_actions,mu)
        if offp == False: 
            a = mdp.getAction(s,n_actions,pi)
        new_s = mdp.getNewState(s,a,P_3D,n_states)
        reward = r_s_a[s,a]
   
        # get the importance ratio
        rho = pi[s,a] / mu[s,a]

        # update values of parameters TD
        tdErrTD = algo.tdError(reward,Phi,s,new_s,thetaTD,discount)
        thetaTD = algo.tdUpdate(thetaTD,tdErrTD,Phi,s,alpha,rho,offp) 
        mspbeTD = mdp.getRMSPBE(Rtarget,Ptarget,Phi,thetaTD,projMatrix,D,discount)
        perf_TD.append(mspbeTD)

        # update values of parameters TDC
        tdErrTDC = algo.tdError(reward,Phi,s,new_s,thetaTDC,discount)
        thetaTDC,wTDC = algo.tdcUpdate(wTDC,thetaTDC,tdErrTDC,Phi,new_s,s,alpha,beta,discount,rho,offp) 
        mspbeTDC = mdp.getRMSPBE(Rtarget,Ptarget,Phi,thetaTDC,projMatrix,D,discount)
        perf_TDC.append(mspbeTDC)

        # iterate
        s = new_s
        count_step += 1
        
    return perf_TDC,perf_TD


##### ON POLICY SETTINGS ####
def onpolicy(alpha,beta,maxSteps,nbRuns1,n_states,n_features,n_actions,P_3D,pi,mu,discount,Phi,Ptarget,D_pi,False,r_s_a) : 

    b = np.linalg.inv((Phi.T).dot(D_pi).dot(Phi))
    projectionMatrixT = Phi.dot(b).dot(Phi.T).dot(D_pi)

    runsTDC = dict()
    runsTD = dict()
    count_run = 0
    while count_run < nbRuns1 : 
        perf_TDC,perf_TD = experimentMDP(maxSteps,n_states,n_features,n_actions,P_3D,pi,mu,discount,Phi,alpha,Ptarget,projectionMatrixT,D_pi,beta,False,r_s_a)
        runsTDC[count_run] = perf_TDC
        runsTD[count_run] = perf_TD
        count_run += 1

    TDC_onp = [0]*maxSteps
    TD_onp = [0]*maxSteps
    for i in range(maxSteps) : 
        for j in range(nbRuns1) : 
            TDC_onp[i] += runsTDC[j][i]
            TD_onp[i] += runsTD[j][i]
            
    TDC_onp = [elt/float(nbRuns1) for elt in TDC_onp]
    TD_onp = [elt/float(nbRuns1) for elt in TD_onp]

    return TDC_onp,TD_onp

######### OFF POLICY SETTINGS ########

def offpolicy(alpha,beta,maxSteps2,nbRuns2,n_states,n_features,n_actions,P_3D,pi,mu,discount,Phi,Ptarget,D_mu,True,r_s_a) : 
    b2 = np.linalg.inv((Phi.T).dot(D_mu).dot(Phi))
    projectionMatrixOP = Phi.dot(b2).dot(Phi.T).dot(D_mu)

    runsTDC2 = dict()
    runsTD2 = dict()
    count_run = 0
    nbRuns = 20
    while count_run < nbRuns : 
        perf_TDC2,perf_TD2 = experimentMDP(maxSteps2,n_states,n_features,n_actions,P_3D,pi,mu,discount,Phi,alpha,Ptarget,projectionMatrixOP,D_mu,beta,True,r_s_a)
        runsTDC2[count_run] = perf_TDC2
        runsTD2[count_run] = perf_TD2
        count_run += 1

    TDC_offp2 = [0]*maxSteps2
    TD_offp2 = [0]*maxSteps2
    for i in range(maxSteps2) : 
        for j in range(nbRuns) : 
            TDC_offp2[i] += runsTDC2[j][i]
            TD_offp2[i] += runsTD2[j][i]
            
    TDC_offp2 = [elt/float(nbRuns) for elt in TDC_offp2]
    TD_offp2 = [elt/float(nbRuns) for elt in TD_offp2]

    return TDC_offp2,TD_offp2















