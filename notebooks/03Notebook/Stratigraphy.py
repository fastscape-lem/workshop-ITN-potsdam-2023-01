import numpy as np
import numba
@numba.njit
def plotStratigraphy(XorY_StratiOverTime,XorY_GrainSizeOverTime):
    i=0
    j=0
    C=np.zeros(((XorY_StratiOverTime.shape[0]),(XorY_StratiOverTime.shape[1])))
    for i in range(0,(XorY_StratiOverTime.shape[1])):
        for j in range(0,(XorY_StratiOverTime.shape[0])):
            tryff=np.array([XorY_GrainSizeOverTime[j,i],XorY_GrainSizeOverTime[j,i+1],XorY_GrainSizeOverTime[j+1,i],XorY_GrainSizeOverTime[j+1,i+1]])
            C[j,i]=np.nanmean(tryff)
    return C

@numba.njit
def plotStratigraphy2(XorY_StratiOverTime,XorY_GrainSizeOverTime):
    i=0
    j=0
    C=np.zeros(((XorY_StratiOverTime.shape[0]),(XorY_StratiOverTime.shape[1])))
    for i in range(0,(XorY_StratiOverTime.shape[1])):
        for j in range(0,(XorY_StratiOverTime.shape[0])):
            #tryff=np.array([XorY_GrainSizeOverTime[j,i],XorY_GrainSizeOverTime[j,i+1],XorY_GrainSizeOverTime[j+1,i],XorY_GrainSizeOverTime[j+1,i+1]])
            C[j,i]=(XorY_GrainSizeOverTime[j,i])
    return C
