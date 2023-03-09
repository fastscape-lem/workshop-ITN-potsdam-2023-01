import numpy as np
import numba

# GSFast Functions Needed:

#1) Calculate the sediment flux by integrating a value (erosion rate) through the drainage network
#
#Multiflow sum_stack 
#!! Due to the nrec, this is different than the single flow sum_stack
@numba.njit
def sum_stack(stack,rec,flatVal,nrec,weight):
    nrec= np.where(nrec>0,nrec,0)
    #st= np.where(stack>0,stack,0)
    EE=flatVal.copy() #!!! COPY
    for i in stack:
        for j in range(nrec[i]):    #if there are multiple recievers, use the weight to determine the flux (later) transfered
            r=rec[i,j]
            r=np.where(r>0,r,0)
            if r==i: #This is another way to avoid negative/undefined recievers
                continue
            EE[r]=EE[r]+(EE[i]*weight[i,j])  #add the erosion rate to itself through the stack to calculate the flux.
    return EE
    
#2) Integrate the Y star through the drainage network weighted by the drainage
    # Routine to compute the ystar and D0 parameters of Paola and Seal 1995 in 2D
@numba.njit
def qD0sum_stack(stack,rec,Rstar,q,nrec,weight,D0):
    '''
    Computes ystar (from Rstar) and D0 (from Paola and Seal (1995) by propagating both quantities
    through the stack order in a multi-direction flow environment using sediment flux values to
    compute weights. This gives for ystar:
    1. Initialize ystar=Rstar
    2. for receiver j of node i (on stack): ystar[r[i,j]]=ystar[r[i,j]]+w[r,i]*ystar[i]
    where w[r,k] is the weight of donor i to node r

    In input:
    stack: stack order (dimension n, where n is number of nodes in the grid
    rec: receiver information (dimension(n,8) as there are 8 potential receivers per node)
    nrec: number of receivers information (dimension n)
    weight: receiver weight information (dimension(n,8) as there are as many weights as receivers)
    flatVal: Rstar (dimension n)
    D0: initial value for D0 (dimension(n)

    In output:
    ystar: ystar (dimension n)
    D0c: updated value of D0 (dimension n)
    '''
# we check for negative number of receivers (happens when a node has no receiver, such as a
# baselevel node or a local minima if they have not been removed
    nrec= np.where(nrec>0,nrec,0)
# we first initialize ndon and wdon, the number of donors per node and the associated weights
    n = len(nrec)
    ndon = np.zeros(n)
    wdon = np.zeros((n,8))
# we compute the donor weight and the number of donors by inverting the receiver weight information
    for i in stack:
        for k in range(nrec[i]):
            r = rec[i,k]
            if r!=i:
                wdon[r,int(ndon[r])] = q[i]*weight[i,k]
                ndon[r] = ndon[r]+1
# we normalize the donor weights so that their sum (for each node) = 1
    for i in range(n):
        if ndon[i]!=0:
            wsum = np.sum(wdon[i,:int(ndon[i])])
            wdon[i,:int(ndon[i])] = wdon[i,:int(ndon[i])]/wsum
# we initialize ystar (=Rstar) and D0c (=D0)
    y = Rstar.copy()
    D0c = D0.copy()
    D0c = np.where(ndon==0, D0c, 0)
# we accumulate ystar and D0 following the same stack order so that we reach the donors
# in the same order as when we computed the donor weights
# this is important for this algorithm to work!!!
    ndon = ndon*0
    for i in stack:
        for k in range(nrec[i]):
            r = rec[i,k]
            if r!=i:
                y[r] = y[r] + y[i]*wdon[r,int(ndon[r])]
                D0c[r] = D0c[r] + D0c[i]*wdon[r,int(ndon[r])]
                ndon[r] = ndon[r]+1
# we return the solution
    return y, D0c







