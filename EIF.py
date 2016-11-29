import numpy as np
from numpy import dot

1./2

#Information matrix H_t
# TODO : Initialize ?
infoMatrix = np.array()

# Information vector b_t
#TODO : initialize ?
infoVector = np.array()

# Observed yi

#hGrad = C_t
#sensorMeasure = z_t
def updateInformation():
    #N = number of known landmarks

    hGrad = np.zeros(N + 1)
    hGrad[0] = #dh/dx
    hGrad[i] = #dh/dyi
    identity = np.eye(N + 1)

    gGrad =
    covariance = np.inv(infoMatrix)

    infoVectorBar = dot(dot(identity + gGrad, covariance), (identity + gGrad).T) + Sx
    infoVectorBar = np.inv(infoVectorBar)

    zInvC = dot(np.inv(Z), hGrad.T)
    infoMatrix = infoMatrixBar +  dot(hGrad,zInvC)
    infoVector = infoVectorBar +  dot(sensorMeasure - h(mu) + dot(hGrad.T, mu.T) ,zInvC)




def

