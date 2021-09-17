import pygame as pg
import os
import pandas as pd
import collections as co
import numpy as np
import pickle
import sys
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..')))
from src.UpdateWorld import *


class Experiment():
    def __init__(self, trial, writer, experimentValues, initialWorld, updateWorld, drawImage, resultsPath, minDistanceBetweenGrids, maxDistanceBetweenGrids, restImage, finishImage, restTrial):
        self.trial = trial
        self.writer = writer
        self.experimentValues = experimentValues
        self.initialWorld = initialWorld
        self.updateWorld = updateWorld
        self.drawImage = drawImage
        self.resultsPath = resultsPath
        self.minDistanceBetweenGrids = minDistanceBetweenGrids
        self.maxDistanceBetweenGrids = maxDistanceBetweenGrids
        self.restImage = restImage
        self.finishImage = finishImage
        self.restTrial = restTrial

    def __call__(self, designValues):
        bean1Grid, bean2Grid, playerGrid, angle = self.initialWorld(self.minDistanceBetweenGrids, self.maxDistanceBetweenGrids)
        trialIndex = 0
        while trialIndex < len(designValues):
            self.experimentValues["angle"] = angle
            results, bean1Grid, playerGrid = self.trial(bean1Grid, bean2Grid, playerGrid)
            response = self.experimentValues.copy()
            response.update(results)
            responseDF = pd.DataFrame(response, index=[trialIndex])
            self.writer(responseDF)
            bean2Grid, nextCondition, angle = self.updateWorld(bean1Grid, playerGrid, designValues[trialIndex])
            self.experimentValues["angle"] = angle
            self.experimentValues["condition"] = nextCondition
            designValues = adjustDesignValues(nextCondition, trialIndex, designValues)
            trialIndex = trialIndex + 1
            if trialIndex in self.restTrial:
                self.drawImage(self.restImage)
        self.drawImage(self.finishImage)


class ModelExperiment():
    def __init__(self, trial, writer, experimentValues, initialWorld, updateWorld, drawImage, resultsPath, minDistanceBetweenGrids, maxDistanceBetweenGrids):
        self.trial = trial
        self.writer = writer
        self.experimentValues = experimentValues
        self.initialWorld = initialWorld
        self.updateWorld = updateWorld
        self.drawImage = drawImage
        self.resultsPath = resultsPath
        self.minDistanceBetweenGrids = minDistanceBetweenGrids
        self.maxDistanceBetweenGrids = maxDistanceBetweenGrids

    def __call__(self, designValues):
        bean1Grid, bean2Grid, playerGrid, angle = self.initialWorld(self.minDistanceBetweenGrids, self.maxDistanceBetweenGrids)
        trialIndex = 0
        while trialIndex < len(designValues):
            self.experimentValues["angle"] = angle
            results, bean1Grid, playerGrid = self.trial(bean1Grid, bean2Grid, playerGrid)
            response = self.experimentValues.copy()
            response.update(results)
            responseDF = pd.DataFrame(response, index=[trialIndex])
            self.writer(responseDF)
            bean2Grid, nextCondition, angle = self.updateWorld(bean1Grid, playerGrid, designValues[trialIndex])
            self.experimentValues["angle"] = angle
            self.experimentValues["condition"] = nextCondition
            designValues = adjustDesignValues(nextCondition, trialIndex, designValues)
            trialIndex = trialIndex + 1


class ModelSimulation():
    def __init__(self, trial, writer, experimentValues, initialWorld, updateWorld, drawImage, resultsPath, minDistanceBetweenGrids, maxDistanceBetweenGrids, runVI):
        self.trial = trial
        self.writer = writer
        self.experimentValues = experimentValues
        self.initialWorld = initialWorld
        self.updateWorld = updateWorld
        self.drawImage = drawImage
        self.resultsPath = resultsPath
        self.minDistanceBetweenGrids = minDistanceBetweenGrids
        self.maxDistanceBetweenGrids = maxDistanceBetweenGrids
        self.runVI = runVI

    def __call__(self, designValues):
        bean1Grid, bean2Grid, playerGrid, angle = self.initialWorld(self.minDistanceBetweenGrids, self.maxDistanceBetweenGrids)
        trialIndex = 0
        while trialIndex < len(designValues):
            self.experimentValues["angle"] = angle
            QDict = self.runVI((bean1Grid, bean2Grid))
            results, bean1Grid, playerGrid = self.trial(bean1Grid, bean2Grid, playerGrid, QDict)
            response = self.experimentValues.copy()
            response.update(results)
            responseDF = pd.DataFrame(response, index=[trialIndex])
            self.writer(responseDF)
            bean2Grid, nextCondition, angle = self.updateWorld(bean1Grid, playerGrid, designValues[trialIndex])
            self.experimentValues["angle"] = angle
            self.experimentValues["condition"] = nextCondition
            designValues = adjustDesignValues(nextCondition, trialIndex, designValues)
            trialIndex = trialIndex + 1
