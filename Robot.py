import GeneratePoints
import math
import numpy as np

class Robot():
    def __init__(self, xMax, yMax):
        self.xMax = xMax
        self.yMax = yMax
        self.position = GeneratePoints(1,xMax,yMax)
        self.direction = 2 * math.pi * np.random.rand()

    def walk(self, direction, distance):
        position = self.position
        self.direction = direction

        position[0] = position[0] + distance * math.cos(direction)
        position[1] = position[1] + distance * math.sin(direction)

        self.position = position