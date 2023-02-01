import numpy as np
import numba


#For when batch is stacked as one dimension 
#import numba
#@numba.njit
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

