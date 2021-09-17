import pygame as pg
import os
import collections as co
import numpy as np
import pickle
import sys
import math
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..')))

from src.Visualization import DrawBackground, DrawNewState, DrawImage
from src.Controller import HumanController, CheckBoundary
from src.UpdateWorld import *
from src.Trial import Trial
from src.Experiment import Experiment
from src.Writer import WriteDataFrameToCSV


def main():
    dimension = 15
    bounds = [0, 0, dimension - 1, dimension - 1]
    condition = [-5, -3, -1, 0, 1, 3, 5]
    minDistanceBetweenGrids = max(condition) + 1
    maxDistanceBetweenGrids = calculateMaxDistanceOfGrid(bounds) - minDistanceBetweenGrids
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

    humanController = HumanController(dimension)
    controller = humanController
    experimentValues = co.OrderedDict()
    # experimentValues["name"] = input("Please enter your name:").capitalize()
    experimentValues["name"] = 'test'

    experimentValues["condition"] = 'None'
    writerPath = resultsPath + experimentValues["name"] + '.csv'
    writer = WriteDataFrameToCSV(writerPath)

    introductionImage = pg.image.load(picturePath + 'introduction.png')
    restImage = pg.image.load(picturePath + 'rest.png')
    finishImage = pg.image.load(picturePath + 'finish.png')
    introductionImage = pg.transform.scale(introductionImage, (screenWidth, screenHeight))
    finishImage = pg.transform.scale(finishImage, (int(screenWidth * 2 / 3), int(screenHeight / 4)))
    drawBackground = DrawBackground(screen, dimension, leaveEdgeSpace, backgroundColor, lineColor, lineWidth, textColorTuple)
    drawNewState = DrawNewState(screen, drawBackground, targetColor, playerColor, targetRadius, playerRadius)
    drawImage = DrawImage(screen)

    block = 15
    designValues = createDesignValues(condition * 3, block)
    checkBoundary = CheckBoundary([0, dimension - 1], [0, dimension - 1])
    trial = Trial(controller, drawNewState, checkBoundary)

    restTrialInterval = math.ceil(len(designValues) / 6)
    restTrial = list(range(0, len(designValues), restTrialInterval))

    experiment = Experiment(trial, writer, experimentValues, initialWorld, updateWorld, drawImage, resultsPath, minDistanceBetweenGrids, maxDistanceBetweenGrids, restImage, finishImage, restTrial)
    drawImage(introductionImage)
    experiment(designValues)
    # print(len(designValues))


if __name__ == "__main__":
    main()
