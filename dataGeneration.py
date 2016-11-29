import numpy as np
import matplotlib.pyplot as plt
from GeneratePoints import GeneratePoints


def testRobot():
    # Create points
    xMax, yMax = 5, 100

    landmarks = GeneratePoints(50, xMax, yMax)
    robot = GeneratePoints(1, xMax, yMax)

    plt.scatter(landmarks[:, 0], landmarks[:, 1])
    plt.scatter(robot[:, 0],  robot[:, 1], color="red")
    plt.show()


testRobot()
