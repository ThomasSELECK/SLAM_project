from GeneratePoints import GeneratePoints
import math
import numpy as np


class Robot:
    def __init__(self, xMax, yMax):
        self.xMax = xMax
        self.yMax = yMax
        self.position = GeneratePoints(1, xMax, yMax)
        self.direction = 2 * math.pi * np.random.rand()
        self.positionThought = np.array([0,0])
        self.directionThought = 0.
        self.detectedPoints = np.array([])

    def walk(self, turnDirection, distance):
        position = self.position
        direction = self.direction
        directionThought = self.directionThought
        positionThought = self.positionThought

        directionThought = (directionThought + turnDirection) % (2 * math.pi)
        positionThought += distance * np.array([math.cos(directionThought), math.sin(directionThought)])

        #TODO: Add error parameter in control
        errorDistance = np.random.normal()
        errorDirection = np.random.normal()

        distance = max(0, distance + errorDistance)
        turnDirection += errorDirection

        direction = (direction + turnDirection) % (2 * math.pi)
        position += distance * np.array([math.cos(direction), math.sin(direction)])

        self.position = position
        self.direction = direction
        self.directionThought = directionThought
        self.positionThought = positionThought

    def look(self, pointsRelativePositions):
        positionThought = self.positionThought
        directionThought = self.directionThought

        pointsRelativeDistance = pointsRelativePositions[:, 0]
        pointsRelativeAngle = pointsRelativePositions[:, 1]

        # TODO: Add error parameter in detection
        pointsRelativeDistance += np.random.normal(size=pointsRelativePositions.shape[0])
        pointsRelativeAngle += np.random.normal(size=pointsRelativePositions.shape[0])

        pointsAngle = directionThought + pointsRelativeAngle

        detectedPoints = positionThought.reshape(1,2) + pointsRelativeDistance * np.array([math.cos(pointsAngle), math.sin(pointsAngle)])

        self.detectedPoints = detectedPoints
