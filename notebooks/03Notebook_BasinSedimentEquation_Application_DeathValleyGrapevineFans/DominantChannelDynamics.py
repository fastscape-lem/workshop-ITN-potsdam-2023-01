import numpy as np
import numba


#For when batch is stacked as one dimension 
#import numba
#@numba.njit

# - Functions AvulsionMainChannel was designed for channels flowing in one direction from a orogenic source either in the x or y direction. 
# - In all the examples above, downstream from the orogenic front is along the x axis. 
# - Channels then migrate/avulse from their pathways along the y axis. 
# - AvulsionMainChannel takes 1)Binary: an empty (filled with zeros) array of the dimensions (eg (non-batch input with 3 dimensions): time, y, and x) of the Fastscape drainge output.
#     2) temp: the location of the dominant drainage pathway for each time step. 
# - AvulsionMainChannel can take batch inputs, but they need to be stacked as one batch input (eg: 4 dimensions). 
# - AvulsionMainChannel outputs a binary grid of where in the x and y the position of the channel changed between time steps (a mobility event).
# - In post processing, this can be summarized and divided by the time steps-1 in quesiton to derive a mobility frequency. 
def AvulsionMainChannel(Binary,temp):
    array=np.shape(Binary)
    if len(array)>=4:
        for a in range(np.shape(Binary)[0]):
            for b in range(np.shape(Binary)[1]):
                for d in range(np.shape(Binary)[3]):
                    Binary[a,b,temp[a,b,d],d]=1;
        AvulsionOccurance=np.zeros_like(Binary)
        for a in range(np.shape(Binary)[0]):
            for b in range(np.shape(Binary)[1]):
                for d in range(np.shape(Binary)[3]):                
                    if b != (np.shape(Binary)[1])-1:
                        #print(Binary[a,b,temp[a,b,d],d])
                        #print(Binary[a,b+1,temp[a,b,d],d])
                        #print('break')
                        AvulsionOccurance[a,b,temp[a,b,d],d]=int(Binary[a,b,temp[a,b,d],d] != Binary[a,b+1,temp[a,b,d],d])
        batchAvulse= AvulsionOccurance
    else:
                #for a in range(np.shape(Binary)[0]):
        for b in range(np.shape(Binary)[0]):
            for d in range(np.shape(Binary)[2]):
                Binary[b,temp[b,d],d]=1;
        AvulsionOccurance=np.zeros_like(Binary)
            #for a in range(np.shape(Binary)[0]):
        for b in range(np.shape(Binary)[0]):
            for d in range(np.shape(Binary)[2]):                
                if b != (np.shape(Binary)[0])-1:
                            #print(Binary[a,b,temp[a,b,d],d])
                            #print(Binary[a,b+1,temp[a,b,d],d])
                            #print('break')
                    AvulsionOccurance[b,temp[b,d],d]=int(Binary[b,temp[b,d],d] != Binary[b+1,temp[b,d],d])
        batchAvulse=  AvulsionOccurance 
    return batchAvulse


# Functions find_slopes_withMinima (h, stack, rec,nrec) takes the drainage stack, recievers, and number of recieves, and topography elevation as inputs (1D arrays- need to turn topography into a 1D array) to 
# calculate the slope between nodes. Where local minima occur (slopes less than or equal to zero), the slope is set to zero. In post-processing you can than calculate where the slope is zero as the local minima locations in the stack. 

#Calculates the slope between nodes within the drainage network where slopes that are zero or negative are set to zero (most likely local minima).
def find_slopes_withMinima (h, stack, rec,nrec):
    #This function computes the slope between each node and its receiver in multiflow
        #h : float #1D array containing landscape height
        #rec : int #1D array containing the list of receivers of each node. rec[ij] contains the list of receivers of ij.    
    nrec= np.where(nrec>0,nrec,0)
    #st= np.where(stack>0,stack,0)
    Slope=np.zeros_like(h) #!!! COPY
    for i in stack:
        r=rec[i,:]
        r=np.where(r>0,r,0)
        if r==i: #This is another way to avoid negative/undefined recievers
               continue
        temp=h[i]-h[r]
        if (any(temp <= 0)==1):
               Slope[i]=0
        else:
            Slope[i]=np.nanmean(temp)
    return Slope

