import numpy as np
import pygame as pg
from pygame import time
import collections as co
import pickle
import random


def inferGoal(originGrid, aimGrid, targetGridA, targetGridB):
    pacmanBean1aimDisplacement = np.linalg.norm(np.array(targetGridA) - np.array(aimGrid), ord=1)
    pacmanBean2aimDisplacement = np.linalg.norm(np.array(targetGridB) - np.array(aimGrid), ord=1)
    pacmanBean1LastStepDisplacement = np.linalg.norm(np.array(targetGridA) - np.array(originGrid), ord=1)
    pacmanBean2LastStepDisplacement = np.linalg.norm(np.array(targetGridB) - np.array(originGrid), ord=1)
    bean1Goal = pacmanBean1LastStepDisplacement - pacmanBean1aimDisplacement
    bean2Goal = pacmanBean2LastStepDisplacement - pacmanBean2aimDisplacement
    if bean1Goal > bean2Goal:
        goal = 1
    elif bean1Goal < bean2Goal:
        goal = 2
    else:
        goal = 0
    return goal


def extractNoRepeatingElements(list, number):
    point = random.sample(list, number)
    return point


def checkTerminationOfTrial(bean1Grid, bean2Grid, humanGrid):
    if np.linalg.norm(np.array(humanGrid) - np.array(bean1Grid), ord=1) == 0 or np.linalg.norm(np.array(humanGrid) - np.array(bean2Grid), ord=1) == 0:
        pause = False
    else:
        pause = True
    return pause


class NormalTrial():
    def __init__(self, controller, drawNewState, drawText, normalNoise, checkBoundary):
        self.controller = controller
        self.drawNewState = drawNewState
        self.drawText = drawText
        self.normalNoise = normalNoise
        self.checkBoundary = checkBoundary

    def __call__(self, bean1Grid, bean2Grid, playerGrid, designValues):
        initialPlayerGrid = playerGrid
        initialTime = time.get_ticks()
        reactionTime = list()
        trajectory = [initialPlayerGrid]
        results = co.OrderedDict()
        aimActionList = list()
        totalStep = int(np.linalg.norm(np.array(playerGrid) - np.array(bean1Grid), ord=1))
        noiseStep = random.sample(list(range(1, totalStep + 1)), designValues)
        stepCount = 0
        goalList = list()

        self.drawText("+", [0, 0, 0], [7, 7])
        pg.time.wait(1300)
        self.drawNewState(bean1Grid, bean2Grid, initialPlayerGrid)
        pg.event.set_allowed([pg.KEYDOWN, pg.KEYUP, pg.QUIT])

        realPlayerGrid = initialPlayerGrid
        self.drawNewState(bean1Grid, bean2Grid, initialPlayerGrid)
        pause = True
        initialTime = time.get_ticks()
        while pause:
            aimPlayerGrid, aimAction = self.controller(realPlayerGrid, bean1Grid, bean2Grid)
            reactionTime.append(time.get_ticks() - initialTime)
            goal = inferGoal(trajectory[-1], aimPlayerGrid, bean1Grid, bean2Grid)
            goalList.append(goal)
            stepCount = stepCount + 1
            noisePlayerGrid, aimAction = self.normalNoise(trajectory[-1], aimAction, trajectory, noiseStep, stepCount)
            realPlayerGrid = self.checkBoundary(noisePlayerGrid)
            self.drawNewState(bean1Grid, bean2Grid, realPlayerGrid)
            trajectory.append(list(realPlayerGrid))
            aimActionList.append(aimAction)
            pause = checkTerminationOfTrial(bean1Grid, bean2Grid, realPlayerGrid)
        pg.time.wait(500)
        pg.event.set_blocked([pg.KEYDOWN, pg.KEYUP])
        results["bean1GridX"] = bean1Grid[0]
        results["bean1GridY"] = bean1Grid[1]
        results["bean2GridX"] = bean2Grid[0]
        results["bean2GridY"] = bean2Grid[1]
        results["playerGridX"] = initialPlayerGrid[0]
        results["playerGridY"] = initialPlayerGrid[1]
        results["reactionTime"] = str(reactionTime)
        results["trajectory"] = str(trajectory)
        results["aimAction"] = str(aimActionList)
        results["noisePoint"] = str(noiseStep)
        results["goal"] = str(goalList)
        return results


class SpecialTrial():
    def __init__(self, controller, drawNewState, drawText, awayFromTheGoalNoise, checkBoundary):
        self.controller = controller
        self.drawNewState = drawNewState
        self.drawText = drawText
        self.awayFromTheGoalNoise = awayFromTheGoalNoise
        self.checkBoundary = checkBoundary

    def __call__(self, bean1Grid, bean2Grid, playerGrid, designValues):
        initialPlayerGrid = playerGrid
        initialTime = time.get_ticks()
        reactionTime = list()
        trajectory = [initialPlayerGrid]
        results = co.OrderedDict()
        aimActionList = list()
        firstIntentionFlag = False
        noiseStep = list()
        stepCount = 0
        goalList = list()
        self.drawText("+", [0, 0, 0], [7, 7])
        pg.time.wait(1300)
        self.drawNewState(bean1Grid, bean2Grid, initialPlayerGrid)
        pg.event.set_allowed([pg.KEYDOWN, pg.KEYUP, pg.QUIT])

        realPlayerGrid = initialPlayerGrid
        pause = True
        initialTime = time.get_ticks()
        while pause:
            aimPlayerGrid, aimAction = self.controller(realPlayerGrid, bean1Grid, bean2Grid)
            reactionTime.append(time.get_ticks() - initialTime)

            goal = inferGoal(trajectory[-1], aimPlayerGrid, bean1Grid, bean2Grid)
            goalList.append(goal)
            stepCount = stepCount + 1
            noisePlayerGrid, firstIntentionFlag, noiseStep = self.awayFromTheGoalNoise(
                trajectory[-1], bean1Grid, bean2Grid, aimAction, goal, firstIntentionFlag, noiseStep, stepCount)
            realPlayerGrid = self.checkBoundary(noisePlayerGrid)
            self.drawNewState(bean1Grid, bean2Grid, realPlayerGrid)
            trajectory.append(list(realPlayerGrid))
            aimActionList.append(aimAction)
            pause = checkTerminationOfTrial(bean1Grid, bean2Grid, realPlayerGrid)
        pg.time.wait(500)
        pg.event.set_blocked([pg.KEYDOWN, pg.KEYUP])
        results["bean1GridX"] = bean1Grid[0]
        results["bean1GridY"] = bean1Grid[1]
        results["bean2GridX"] = bean2Grid[0]
        results["bean2GridY"] = bean2Grid[1]
        results["playerGridX"] = initialPlayerGrid[0]
        results["playerGridY"] = initialPlayerGrid[1]
        results["reactionTime"] = str(reactionTime)
        results["trajectory"] = str(trajectory)
        results["aimAction"] = str(aimActionList)
        results["noisePoint"] = str(noiseStep)
        results["goal"] = str(goalList)
        return results


class NormalTrialWithGoal():
    def __init__(self, controller, drawNewState, drawText, normalNoise, checkBoundary, initPrior, inferGoalPosterior):
        self.controller = controller
        self.drawNewState = drawNewState
        self.drawText = drawText
        self.normalNoise = normalNoise
        self.checkBoundary = checkBoundary
        self.initPrior = initPrior
        self.inferGoalPosterior = inferGoalPosterior

    def checkTerminationOfTrial(self, bean1Grid, bean2Grid, humanGrid):
        if np.linalg.norm(np.array(humanGrid) - np.array(bean1Grid), ord=1) == 0 or \
                np.linalg.norm(np.array(humanGrid) - np.array(bean2Grid), ord=1) == 0:
            pause = False
        else:
            pause = True
        return pause

    def __call__(self, bean1Grid, bean2Grid, playerGrid, designValues):
        initialPlayerGrid = playerGrid
        initialTime = time.get_ticks()
        reactionTime = list()
        trajectory = [initialPlayerGrid]
        results = co.OrderedDict()
        aimActionList = list()
        totalStep = int(np.linalg.norm(np.array(playerGrid) - np.array(bean1Grid), ord=1))
        noiseStep = random.sample(list(range(1, totalStep + 1)), designValues)
        stepCount = 0
        goalList = list()

        # self.drawText("+", [0, 0, 0], [7, 7])
        # pg.time.wait(1300)
        # self.drawNewState(bean1Grid, bean2Grid, initialPlayerGrid)
        # pg.event.set_allowed([pg.KEYDOWN, pg.KEYUP, pg.QUIT])

        priorList = self.initPrior
        realPlayerGrid = initialPlayerGrid
        pause = True
        while pause:
            aimPlayerGrid, aimAction = self.controller(realPlayerGrid, bean1Grid, bean2Grid, priorList)
            posteriorList = self.inferGoalPosterior(realPlayerGrid, aimAction, bean1Grid, bean2Grid, priorList)
            priorList = posteriorList

            goal = inferGoal(trajectory[-1], aimPlayerGrid, bean1Grid, bean2Grid)
            goalList.append(goal)
            stepCount = stepCount + 1
            noisePlayerGrid, realAction = self.normalNoise(trajectory[-1], aimAction, trajectory, noiseStep, stepCount)
            realPlayerGrid = self.checkBoundary(noisePlayerGrid)
            # self.drawNewState(bean1Grid, bean2Grid, realPlayerGrid)
            # pg.time.delay(1000)
            reactionTime.append(time.get_ticks() - initialTime)
            trajectory.append(list(realPlayerGrid))
            aimActionList.append(aimAction)
            pause = self.checkTerminationOfTrial(bean1Grid, bean2Grid, realPlayerGrid)
        # pg.time.wait(500)
        pg.event.set_blocked([pg.KEYDOWN, pg.KEYUP])
        results["bean1GridX"] = bean1Grid[0]
        results["bean1GridY"] = bean1Grid[1]
        results["bean2GridX"] = bean2Grid[0]
        results["bean2GridY"] = bean2Grid[1]
        results["playerGridX"] = initialPlayerGrid[0]
        results["playerGridY"] = initialPlayerGrid[1]
        results["reactionTime"] = str(reactionTime)
        results["trajectory"] = str(trajectory)
        results["aimAction"] = str(aimActionList)
        results["noisePoint"] = str(noiseStep)
        results["goal"] = str(goalList)
        return results


class SpecialTrialWithGoal():
    def __init__(self, controller, drawNewState, drawText, awayFromTheGoalNoise, checkBoundary, initPrior, inferGoalPosterior):
        self.controller = controller
        self.drawNewState = drawNewState
        self.drawText = drawText
        self.awayFromTheGoalNoise = awayFromTheGoalNoise
        self.checkBoundary = checkBoundary
        self.initPrior = initPrior
        self.inferGoalPosterior = inferGoalPosterior

    def checkTerminationOfTrial(self, bean1Grid, bean2Grid, humanGrid):
        if np.linalg.norm(np.array(humanGrid) - np.array(bean1Grid), ord=1) == 0 or \
                np.linalg.norm(np.array(humanGrid) - np.array(bean2Grid), ord=1) == 0:
            pause = False
        else:
            pause = True
        return pause

    def __call__(self, bean1Grid, bean2Grid, playerGrid, designValues):
        initialPlayerGrid = playerGrid
        initialTime = time.get_ticks()
        reactionTime = list()
        trajectory = [initialPlayerGrid]
        results = co.OrderedDict()
        aimActionList = list()
        firstIntentionFlag = False
        noiseStep = list()
        stepCount = 0
        goalList = list()

        # self.drawText("+", [0, 0, 0], [7, 7])
        # pg.time.wait(1300)
        # self.drawNewState(bean1Grid, bean2Grid, initialPlayerGrid)
        pg.event.set_allowed([pg.KEYDOWN, pg.KEYUP, pg.QUIT])

        priorList = self.initPrior
        realPlayerGrid = initialPlayerGrid
        pause = True
        while pause:
            aimPlayerGrid, aimAction = self.controller(realPlayerGrid, bean1Grid, bean2Grid, priorList)
            posteriorList = self.inferGoalPosterior(realPlayerGrid, aimAction, bean1Grid, bean2Grid, priorList)
            priorList = posteriorList

            goal = inferGoal(trajectory[-1], aimPlayerGrid, bean1Grid, bean2Grid)
            goalList.append(goal)
            stepCount = stepCount + 1
            noisePlayerGrid, firstIntentionFlag, noiseStep = self.awayFromTheGoalNoise(
                trajectory[-1], bean1Grid, bean2Grid, aimAction, goal, firstIntentionFlag,
                noiseStep, stepCount)
            realPlayerGrid = self.checkBoundary(noisePlayerGrid)
            # self.drawNewState(bean1Grid, bean2Grid, realPlayerGrid)
            # pg.time.delay(1000)
            reactionTime.append(time.get_ticks() - initialTime)
            trajectory.append(list(realPlayerGrid))
            aimActionList.append(aimAction)
            pause = checkTerminationOfTrial(bean1Grid, bean2Grid, realPlayerGrid)
        # pg.time.wait(500)
        pg.event.set_blocked([pg.KEYDOWN, pg.KEYUP])
        results["bean1GridX"] = bean1Grid[0]
        results["bean1GridY"] = bean1Grid[1]
        results["bean2GridX"] = bean2Grid[0]
        results["bean2GridY"] = bean2Grid[1]
        results["playerGridX"] = initialPlayerGrid[0]
        results["playerGridY"] = initialPlayerGrid[1]
        results["reactionTime"] = str(reactionTime)
        results["trajectory"] = str(trajectory)
        results["aimAction"] = str(aimActionList)
        results["noisePoint"] = str(noiseStep)
        results["goal"] = str(goalList)
        return results
