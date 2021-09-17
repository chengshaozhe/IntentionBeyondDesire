import random
import numpy as np
import copy
from collections import Counter
import math
import matplotlib.pyplot as plt


def computeAngleBetweenTwoVectors(vector1, vector2):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    lenthOfVector1 = np.sqrt(vector1.dot(vector1))
    lenthOfVector2 = np.sqrt(vector2.dot(vector2))
    cosAngle = vector1.dot(vector2) / (lenthOfVector1 * lenthOfVector2)
    if cosAngle < -1:
        cosAngle = -1
    elif cosAngle > 1:
        cosAngle = 1
    angle = math.degrees(np.arccos(cosAngle))
    return angle


def indexCertainNumberInList(list, number):
    indexList = [i for i in range(len(list)) if list[i] == number]
    return indexList


class InitialWorld():
    def __init__(self, bounds):
        self.bounds = bounds

    def __call__(self, minDistanceBetweenGrids, maxDistanceBerweenGrids):
        playerGrid = (random.randint(self.bounds[0], self.bounds[2]),
                      random.randint(self.bounds[1], self.bounds[3]))
        [meshGridX, meshGridY] = np.meshgrid(range(self.bounds[0], self.bounds[2] + 1, 1),
                                             range(self.bounds[1], self.bounds[3] + 1, 1))
        distanceOfPlayerGrid = abs(meshGridX - playerGrid[0]) + abs(meshGridY - playerGrid[1])
        target1GridArea = np.where(distanceOfPlayerGrid > minDistanceBetweenGrids)
        target1GridIndex = random.randint(0, len(target1GridArea[0]) - 1)
        target1Grid = tuple([meshGridX[target1GridArea[0][target1GridIndex]][target1GridArea[1][target1GridIndex]], meshGridY[target1GridArea[0][target1GridIndex]][target1GridArea[1][target1GridIndex]]])
        distanceOfTarget1Grid = abs(meshGridX - target1Grid[0]) + abs(meshGridY - target1Grid[1])
        target2GridArea = np.where((distanceOfPlayerGrid > minDistanceBetweenGrids) * (distanceOfTarget1Grid > minDistanceBetweenGrids) * (distanceOfTarget1Grid < maxDistanceBerweenGrids) == True)
        target2GridIndex = random.randint(0, len(target2GridArea[0]) - 1)
        target2Grid = tuple([meshGridX[target2GridArea[0][target2GridIndex]][target2GridArea[1][target2GridIndex]], meshGridY[target2GridArea[0][target2GridIndex]][target2GridArea[1][target2GridIndex]]])
        vectorBetweenTarget1AndPlayer = np.array(target2Grid) - np.array(playerGrid)
        vectorBetweenTarget2AndPlayer = np.array(target1Grid) - np.array(playerGrid)
        angle = computeAngleBetweenTwoVectors(vectorBetweenTarget1AndPlayer, vectorBetweenTarget2AndPlayer)
        return target1Grid, target2Grid, playerGrid, angle


class UpdateWorld():
    def __init__(self, bounds, conditon, minDistance, maxDistance):
        self.condition = conditon
        self.bounds = bounds
        self.minDistance = minDistance
        self.maxDistance = maxDistance

    def __call__(self, oldTargetGrid, playerGrid, designValue):
        condition = copy.deepcopy(self.condition)
        pause = True
        while pause:
            if len(condition) != 0:
                if designValue not in condition:
                    nextCondition = np.random.choice(condition)
                else:
                    nextCondition = designValue
                distance = np.linalg.norm(np.array(oldTargetGrid) - np.array(playerGrid), ord=1) + nextCondition
                [meshGridX, meshGridY] = np.meshgrid(range(self.bounds[0], self.bounds[2] + 1, 1),
                                                     range(self.bounds[1], self.bounds[3] + 1, 1))
                distanceOfPlayerGrid = abs(meshGridX - playerGrid[0]) + abs(meshGridY - playerGrid[1])
                distanceOfOldTargetGrid = abs(meshGridX - oldTargetGrid[0]) + abs(meshGridY - oldTargetGrid[1])
                newTargetGridAreaIndex = np.where((distanceOfPlayerGrid == distance) * (
                    distanceOfOldTargetGrid >= self.minDistance) * (
                    distanceOfOldTargetGrid <= self.maxDistance) == True)
                if len(newTargetGridAreaIndex[0]) != 0 and distance != 0:
                    newTargetGridArea = [[meshGridX[newTargetGridAreaIndex[0][index]][newTargetGridAreaIndex[1][index]],
                                          meshGridY[newTargetGridAreaIndex[0][index]][newTargetGridAreaIndex[1][index]]] for index in range(len(newTargetGridAreaIndex[0]))]
                    vectorBetweenNewTargetAndPlayer = [np.array(target) - np.array(playerGrid) for target in newTargetGridArea]
                    vectorBetweenOldTargetAndPlayer = np.array(oldTargetGrid) - np.array(playerGrid)
                    angle = [computeAngleBetweenTwoVectors(vector, vectorBetweenOldTargetAndPlayer) for vector in
                             vectorBetweenNewTargetAndPlayer]
                    try:
                        maxAngleIndex = indexCertainNumberInList(angle, max(angle))
                        gridIndex = np.random.choice(maxAngleIndex, 1)
                        newTargetGrid = tuple(newTargetGridArea[gridIndex[0]])
                        pause = False
                        maxAngle = max(angle)
                    except ValueError as e:
                        print(playerGrid, oldTargetGrid, angle)
                else:
                    if len(condition) != 0:
                        condition.remove(nextCondition)

            else:
                newTargetGridAreaIndex = np.where((distanceOfOldTargetGrid > self.minDistance) * (
                    distanceOfPlayerGrid + self.minDistance < self.maxDistance) == True)
                newTargetGridArea = [[meshGridX[newTargetGridAreaIndex[0][index]][newTargetGridAreaIndex[1][index]],
                                      meshGridY[newTargetGridAreaIndex[0][index]][newTargetGridAreaIndex[1][index]]]
                                     for index in range(len(newTargetGridAreaIndex[0]))]
                newTargetGrid = newTargetGridArea[random.randint(0, len(newTargetGridArea) - 1)]
                vectorBetweenNewTargetAndPlayer = np.array(newTargetGrid) - np.array(playerGrid)
                vectorBetweenOldTargetAndPlayer = np.array(oldTargetGrid) - np.array(playerGrid)
                maxAngle = computeAngleBetweenTwoVectors(vectorBetweenNewTargetAndPlayer, vectorBetweenOldTargetAndPlayer)
                nextCondition = "None"
                pause = False
        return newTargetGrid, nextCondition, maxAngle


def calculateMaxDistanceOfGrid(bounds):
    [meshGridX, meshGridY] = np.meshgrid(range(bounds[0], bounds[2] + 1, 1),
                                         range(bounds[1], bounds[3] + 1, 1))
    allDistance = np.array([abs(meshGridX - bounds[0]) + abs(meshGridY - bounds[1]),
                            abs(meshGridX - bounds[2]) + abs(meshGridY - bounds[1]),
                            abs(meshGridX - bounds[0]) + abs(meshGridY - bounds[3]),
                            abs(meshGridX - bounds[2]) + abs(meshGridY - bounds[3])])
    maxDistance = np.min(allDistance.max(0))
    return maxDistance


def createDesignValues(condition, blockNumber):
    designValues = list()
    for block in range(blockNumber):
        random.shuffle(condition)
        designValues.append(condition)
    designValues = np.array(designValues).flatten().tolist()
    return designValues

# def adjustDesignValues(realCondition,aimConditionIndex,designValues):
#     designValuesNotAdjust=np.array(designValues)[:aimConditionIndex]
#     designValuesToAdjust=np.array(designValues)[aimConditionIndex:]
#     newDesignValues=list()
#     newDesignValues=newDesignValues+designValuesNotAdjust.tolist()
#     aimCondition=designValues[aimConditionIndex]
#     a=Counter(designValues)
#     print(a)
#     if realCondition==aimCondition:
#         newDesignValues=designValues
#     else:
#         realConditionIndex=np.where( designValuesToAdjust == realCondition )
#         if len(realConditionIndex)!=0:
#             t=realConditionIndex[0]
#             designValuesToAdjust[realConditionIndex[0]]=aimCondition
#             designValuesToAdjust[0]=realCondition
#             newDesignValues=newDesignValues+designValuesToAdjust.tolist()
#         else:
#             designValuesToAdjust[0]=realCondition
#             designValues=designValuesToAdjust.tolist()
#             designValues.append(aimCondition)
#             newDesignValues = newDesignValues + designValuesToAdjust
#     return newDesignValues


def adjustDesignValues(realCondition, aimConditionIndex, designValues):
    designValuesNotAdjust = np.array(designValues)[:aimConditionIndex]
    designValuesToAdjust = np.array(designValues)[aimConditionIndex:]
    newDesignValues = list()
    newDesignValues = newDesignValues + designValuesNotAdjust.tolist()
    aimCondition = designValues[aimConditionIndex]
    a = Counter(designValues)
    if realCondition == aimCondition:
        newDesignValues = designValues
    else:
        designValuesToAdjust = designValuesToAdjust.tolist()
        try:
            realConditionIndex = designValuesToAdjust.index(realCondition)
            designValuesToAdjust[realConditionIndex] = aimCondition
            designValuesToAdjust[0] = realCondition
            newDesignValues = newDesignValues + designValuesToAdjust
        except ValueError:
            designValuesToAdjust[0] = realCondition
            designValues.append(aimCondition)
            newDesignValues = newDesignValues + designValuesToAdjust
    return newDesignValues
