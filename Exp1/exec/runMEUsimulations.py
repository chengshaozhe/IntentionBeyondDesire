import pygame as pg
import os
import pandas as pd
import collections as co
import numpy as np
import pickle
import random
import os
import sys
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..')))
from itertools import permutations

from src.visualization import DrawBackground, DrawNewState, DrawImage, DrawText
from src.controller import HumanController, ModelController, ModelControllerWithGoal, NormalNoise, AwayFromTheGoalNoise, CheckBoundary, InferGoalPosterior, SoftmaxPolicy
import src.updateWorld
from src.writer import WriteDataFrameToCSV
from src.simulationTrial import NormalTrial, SpecialTrial, NormalTrialWithGoal, SpecialTrialWithGoal


class Experiment():
    def __init__(self, normalTrial, specialTrial, writer, experimentValues, updateWorld, drawImage, resultsPath,
                 minDistanceBetweenGrids):
        self.normalTrial = normalTrial
        self.specialTrial = specialTrial
        self.writer = writer
        self.experimentValues = experimentValues
        self.updateWorld = updateWorld
        self.drawImage = drawImage
        self.resultsPath = resultsPath
        self.minDistanceBetweenGrids = minDistanceBetweenGrids

    def __call__(self, noiseDesignValues, shapeDesignValues):
        for trialIndex in range(len(noiseDesignValues)):
            playerGrid, bean1Grid, bean2Grid, direction = self.updateWorld(shapeDesignValues[trialIndex][0], shapeDesignValues[trialIndex][1])
            if isinstance(noiseDesignValues[trialIndex], int):
                results = self.normalTrial(bean1Grid, bean2Grid, playerGrid, noiseDesignValues[trialIndex])
            else:
                results = self.specialTrial(bean1Grid, bean2Grid, playerGrid, noiseDesignValues[trialIndex])
            results["noiseNumber"] = noiseDesignValues[trialIndex]
            results["bottom"] = shapeDesignValues[trialIndex][0]
            results["height"] = shapeDesignValues[trialIndex][1]
            results["direction"] = direction
            response = self.experimentValues.copy()
            response.update(results)
            responseDF = pd.DataFrame(response, index=[trialIndex])
            self.writer(responseDF)


def main():
    dimension = 15
    minDistanceBetweenGrids = 5
    blockNumber = 3
    noiseCondition = list(permutations([1, 2, 0], 3))
    noiseCondition.append((1, 1, 1))
    picturePath = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/pictures/'
    resultsPath = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/results/'
    machinePolicyPath = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/machinePolicy/'
    bottom = [4, 6, 8]
    height = [6, 7, 8]
    direction = [0, 90, 180, 270]
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

    introductionImage = pg.image.load(picturePath + 'introduction.png')
    finishImage = pg.image.load(picturePath + 'finish.png')
    introductionImage = pg.transform.scale(introductionImage, (screenWidth, screenHeight))
    finishImage = pg.transform.scale(finishImage, (int(screenWidth * 2 / 3), int(screenHeight / 4)))
    drawBackground = DrawBackground(screen, dimension, leaveEdgeSpace, backgroundColor, lineColor, lineWidth, textColorTuple)
    drawText = DrawText(screen, drawBackground)
    drawNewState = DrawNewState(screen, drawBackground, targetColor, playerColor, targetRadius, playerRadius)
    drawImage = DrawImage(screen)

    softmaxBeta = 2.5
    policy = pickle.load(open(machinePolicyPath + "noise0.1commitSTGird15_policy.pkl", "rb"))

    for i in range(50):
        print(i)
        noiseDesignValues = UpdateWorld.createNoiseDesignValue(noiseCondition, blockNumber)
        shapeDesignValues = UpdateWorld.createShapeDesignValue(bottom, height)
        updateWorld = UpdateWorld.UpdateWorld(direction, dimension)

        experimentValues = co.OrderedDict()
        experimentValues["name"] = "softmaxBeta" + str(softmaxBeta) + '_' + str(i)
        resultsDirPath = os.path.join(resultsPath, "softmaxBeta" + str(softmaxBeta))
        if not os.path.exists(resultsDirPath):
            os.makedirs(resultsDirPath)
        writerPath = os.path.join(resultsDirPath, experimentValues["name"] + '.csv')
        writer = WriteDataFrameToCSV(writerPath)

        modelController = ModelController(policy, dimension, softmaxBeta)
        modelController = ModelControllerOnline(softmaxBeta)

        checkBoundary = CheckBoundary([0, dimension - 1], [0, dimension - 1])
        controller = modelControllerWithGoal
        normalNoise = NormalNoise(controller)
        awayFromTheGoalNoise = AwayFromTheGoalNoise(controller)
        normalTrial = NormalTrial(controller, drawNewState, drawText, normalNoise, checkBoundary)
        specialTrial = SpecialTrial(controller, drawNewState, drawText, awayFromTheGoalNoise, checkBoundary)

        experiment = Experiment(normalTrial, specialTrial, writer, experimentValues, updateWorld, drawImage, resultsPath, minDistanceBetweenGrids)
        # debug
        # noiseDesignValues = ['special'] * 10
        experiment(noiseDesignValues, shapeDesignValues)


if __name__ == "__main__":
    main()
