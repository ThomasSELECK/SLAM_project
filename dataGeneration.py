import numpy as np
import matplotlib.pyplot as plt


def generatePoints(nbPoints, xMax, yMax):
    points = (np.random.rand(nbPoints, 2) * 2 - 1)
    points[:, 0] = xMax * points[:, 0]
    points[:, 1] = yMax * points[:, 1]
    return (points)

# Create points
xMax, yMax = 5, 100

landmarks = generatePoints(50, xMax, yMax)

robot = generatePoints(1, xMax, yMax)

plt.scatter(landmarks[:, 0], landmarks[:, 1])
plt.scatter(robot[:, 0],  robot[:, 1], color="red")
plt.show()
