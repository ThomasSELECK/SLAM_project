import numpy as np

def GeneratePoints(nbPoints, xMax, yMax):
    points = (np.random.rand(nbPoints, 2) * 2 - 1)
    points[:, 0] = xMax * points[:, 0]
    points[:, 1] = yMax * points[:, 1]
    return(points)
