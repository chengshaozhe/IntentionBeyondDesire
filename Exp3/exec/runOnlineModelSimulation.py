import pygame as pg
import os
import collections as co
import numpy as np
import pickle
import sys
import math
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..')))

from src.Visualization import DrawBackground, DrawNewState, DrawImage
from src.Controller import ModelControllerOnline, CheckBoundary
from src.UpdateWorld import *
from src.Trial import SimulationTrial
from src.Experiment import ModelSimulation
from src.Writer import WriteDataFrameToCSV
from machinePolicy.onlineVI import RunVI


def main():
    gridSize = dimension = 15
    bounds = [0, 0, dimension - 1, dimension - 1]
    condition = [-5, -3, -1, 0, 1, 3, 5]
    minDistanceBetweenGrids = max(condition) + 1
    maxDistanceBetweenGrids = calculateMaxDistanceOfGrid(bounds) - minDistanceBetweenGrids
    block = 15
    initialWorld = InitialWorld(bounds)
    updateWorld = UpdateWorld(bounds, condition, minDistanceBetweenGrids, maxDistanceBetweenGrids)
    pg.init()
    screenWidth = 680
    screenHeight = 680
    screen = pg.display.set_mode((screenWidth, screenHeight))
    leaveEdgeSpace = 2
    lineWidth = 1
    backgroundColor = [205, 255, 204]
    lineColor = [0, 0, 0]
    targetColor = [255, 50, 50]
    playerColor = [50, 50, 255]
    targetRadius = 10
    playerRadius = 10
    textColorTuple = (255, 50, 50)
    pg.event.set_allowed([pg.KEYDOWN, pg.QUIT])

    picturePath = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/pictures/'
    resultsPath = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/results/'
    drawBackground = DrawBackground(screen, dimension, leaveEdgeSpace, backgroundColor, lineColor, lineWidth, textColorTuple)
    checkBoundary = CheckBoundary([0, dimension - 1], [0, dimension - 1])
    drawNewState = DrawNewState(screen, drawBackground, targetColor, playerColor, targetRadius, playerRadius)
    drawImage = DrawImage(screen)

    softmaxBeta = 1
    noise = 0
    gamma = 0.9
    goalReward = [30, 30]
    actionSpace = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    noiseActionSpace = [(0, 0)]
    runVI = RunVI(gridSize, actionSpace, noiseActionSpace, noise, gamma, goalReward)

    numberOfMachineRun = 20
    for i in range(numberOfMachineRun):
        print(i)
        renderOn = 0
        modelController = ModelControllerOnline(softmaxBeta)
        trial = SimulationTrial(modelController, drawNewState, checkBoundary, renderOn)

        experimentValues = co.OrderedDict()
        experimentValues["name"] = "softmaxBeta" + str(softmaxBeta) + '_' + str(i)
        resultsDirPath = os.path.join(resultsPath, 'reward_' + str(goalReward[0]) + "noise" + str(noise) + '_' + "softmaxBeta" + str(softmaxBeta))

        if not os.path.exists(resultsDirPath):
            os.mkdir(resultsDirPath)
        experimentValues["condition"] = 'None'
        writerPath = os.path.join(resultsDirPath, experimentValues["name"] + '.csv')
        writer = WriteDataFrameToCSV(writerPath)

        experiment = ModelSimulation(trial, writer, experimentValues, initialWorld, updateWorld, drawImage, resultsPath, minDistanceBetweenGrids, maxDistanceBetweenGrids, runVI)

        designValues = createDesignValues(condition * 3, block)
        experiment(designValues)


if __name__ == "__main__":
    main()
