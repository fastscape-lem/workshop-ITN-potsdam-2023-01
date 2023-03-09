import numpy as np
import numba

# - plotStratigraphy takes 1) XorY_StratiOverTime (time and either x or y dimensions): strati__elevation selected for only the basin area and either averaged or selected for one across (y)/down(x) basin distance 
#     2) XorY_GrainSizeOverTime (time and either x or y dimensions) the grain size or erosion rate or other desired variable that will be used to fill the stratigraphy. This also needs to be selected or averaged for one x/y distance. 
# - stratigraphy as it is written assumes that channels are draining either in the x or y direction (mountain along one axis) and stratigraphy is generated along one axis.     
# - plotStratigraphy averages the nearby nodes (grain size or erosion rate or other desired value passed) to fill a given cell of stratigraphy.
# -plotStraigraphy2 does not average the nearest nodes and takes the first closest value to fill the stratigraphy. 

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
