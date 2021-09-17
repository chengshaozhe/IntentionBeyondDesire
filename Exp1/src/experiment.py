import pygame as pg
import os
import pandas as pd
import collections as co
import numpy as np
import pickle
from itertools import permutations
from random import shuffle, choice


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
