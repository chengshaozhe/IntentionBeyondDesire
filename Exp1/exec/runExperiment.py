import pygame as pg
import os
import pandas as pd
import collections as co
import numpy as np
import pickle
from random import shuffle, choice
from itertools import permutations
import sys
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..')))

from src.visualization import DrawBackground, DrawNewState, DrawImage, DrawText
from src.controller import HumanController, ModelController, NormalNoise, AwayFromTheGoalNoise, CheckBoundary
from src.updateWorld import *
from src.writer import WriteDataFrameToCSV
from src.trial import NormalTrial, SpecialTrial
from src.experiment import Experiment


def main():
    dimension = 15
    minDistanceBetweenGrids = 5

    picturePath = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/Pictures/'
    resultsPath = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/Results/'

    bottom = [4, 6, 8]
    height = [6, 7, 8]
    shapeDesignValues = createShapeDesignValue(bottom, height)
    noiseCondition = list(permutations([1, 2, 0], 3))
    noiseCondition.append((1, 1, 1))
    blockNumber = 3
    noiseDesignValues = createNoiseDesignValue(noiseCondition, blockNumber)
    direction = [0, 90, 180, 270]
    updateWorld = UpdateWorld(direction, dimension)

    pg.init()
    screenWidth = 600
    screenHeight = 600
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

    experimentValues = co.OrderedDict()
    experimentValues["name"] = 'test'
    # experimentValues["name"] = input("Please enter your name:").capitalize()
    writerPath = resultsPath + experimentValues["name"] + '.csv'
    writer = WriteDataFrameToCSV(writerPath)

    introductionImage = pg.image.load(picturePath + 'introduction.png')
    finishImage = pg.image.load(picturePath + 'finish.png')
    introductionImage = pg.transform.scale(introductionImage, (screenWidth, screenHeight))
    finishImage = pg.transform.scale(finishImage, (int(screenWidth * 2 / 3), int(screenHeight / 4)))
    drawBackground = DrawBackground(screen, dimension, leaveEdgeSpace, backgroundColor, lineColor, lineWidth,
                                    textColorTuple)
    drawText = DrawText(screen, drawBackground)
    drawNewState = DrawNewState(screen, drawBackground, targetColor, playerColor, targetRadius, playerRadius)
    drawImage = DrawImage(screen)

    humanController = HumanController(dimension)
    checkBoundary = CheckBoundary([0, dimension - 1], [0, dimension - 1])
    controller = humanController
    normalNoise = NormalNoise(controller)
    awayFromTheGoalNoise = AwayFromTheGoalNoise(controller)
    normalTrial = NormalTrial(controller, drawNewState, drawText, normalNoise, checkBoundary)
    specialTrial = SpecialTrial(controller, drawNewState, drawText, awayFromTheGoalNoise, checkBoundary)
    experiment = Experiment(normalTrial, specialTrial, writer, experimentValues, updateWorld, drawImage, resultsPath, minDistanceBetweenGrids)
    drawImage(introductionImage)
    experiment(noiseDesignValues, shapeDesignValues)
    drawImage(finishImage)


if __name__ == "__main__":
    main()
