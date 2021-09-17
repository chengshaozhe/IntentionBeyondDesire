import numpy as np
import pygame as pg
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


def countCertainNumberInList(listToManipulate, certainNumber):
    count = 0
    indexList = list()
    for i in range(len(listToManipulate)):
        if certainNumber == listToManipulate[i]:
            count = count + 1
            indexList.append(i)

    return count, indexList


def calculateSoftmaxProbability(acionValues, beta):
    expont = [min(700, i) for i in np.multiply(beta, acionValues)]
    newProbabilityList = list(np.divide(np.exp(expont), np.sum(np.exp(expont))))

    return newProbabilityList


class SoftmaxPolicy:
    def __init__(self, Q_dict, softmaxBeta):
        self.Q_dict = Q_dict
        self.softmaxBeta = softmaxBeta

    def __call__(self, playerGrid, target1):
        actionDict = self.Q_dict[(playerGrid, target1)]
        actionValues = list(actionDict.values())
        softmaxProbabilityList = calculateSoftmaxProbability(actionValues, self.softmaxBeta)
        softMaxActionDict = dict(zip(actionDict.keys(), softmaxProbabilityList))
        return softMaxActionDict


class NormalNoiseWithDiffSize():
    def __init__(self, noiseSize, controller):
        self.actionSpace = controller.actionSpace
        self.gridSize = controller.gridSize
        self.noiseSize = noiseSize

    def __call__(self, playerGrid, action, trajectory, noiseStep, stepCount):
        if stepCount in noiseStep:
            actionSpace = self.actionSpace.copy()
            actionSpace.remove(action)
            actionList = [str(action) for action in actionSpace]
            actionStr = np.random.choice(actionList)
            realAction = eval(actionStr)
            realAction = tuple(np.array(realAction) * self.noiseSize)
        else:
            realAction = action
        realPlayerGrid = tuple(np.add(playerGrid, realAction))
        return realPlayerGrid, realAction


class NormalNoise():
    def __init__(self, controller):
        self.actionSpace = controller.actionSpace
        self.gridSize = controller.gridSize

    def __call__(self, playerGrid, action, trajectory, noiseStep, stepCount):
        if stepCount in noiseStep:
            actionSpace = self.actionSpace.copy()
            actionSpace.remove(action)
            actionList = [str(action) for action in actionSpace]
            actionStr = np.random.choice(actionList)
            realAction = eval(actionStr)
        else:
            realAction = action
        realPlayerGrid = tuple(np.add(playerGrid, realAction))
        return realPlayerGrid, realAction


def selectActionMinDistanceFromTarget(goal, playerGrid, bean1Grid, bean2Grid, actionSpace):
    allPosiibilePlayerGrid = [np.add(playerGrid, action) for action in actionSpace]
    allActionGoal = [inferGoal(playerGrid, possibleGrid, bean1Grid, bean2Grid) for possibleGrid in
                     allPosiibilePlayerGrid]
    if goal == 1:
        realActionIndex = allActionGoal.index(2)
    else:
        realActionIndex = allActionGoal.index(1)
    realAction = actionSpace[realActionIndex]
    return realAction


class AwayFromTheGoalNoise():
    def __init__(self, controller):
        self.actionSpace = controller.actionSpace
        self.gridSize = controller.gridSize

    def __call__(self, playerGrid, bean1Grid, bean2Grid, action, goal, firstIntentionFlag, noiseStep, stepCount):
        if goal != 0 and not firstIntentionFlag:
            noiseStep.append(stepCount)
            firstIntentionFlag = True
            realAction = selectActionMinDistanceFromTarget(goal, playerGrid, bean1Grid, bean2Grid, self.actionSpace)
        else:
            realAction = action
        realPlayerGrid = tuple(np.add(playerGrid, realAction))
        return realPlayerGrid, firstIntentionFlag, noiseStep


class HumanController():
    def __init__(self, gridSize):
        self.actionDict = {pg.K_UP: (0, -1), pg.K_DOWN: (0, 1), pg.K_LEFT: (-1, 0), pg.K_RIGHT: (1, 0)}
        self.actionSpace = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        self.gridSize = gridSize

    def __call__(self, playerGrid, targetGrid1, targetGrid2):
        action = [0, 0]
        pause = True
        while pause:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        pg.quit()
                        exit()
                if event.type == pg.KEYDOWN:
                    if event.key in self.actionDict.keys():
                        action = self.actionDict[event.key]
                        aimePlayerGrid = tuple(np.add(playerGrid, action))
                        pause = False
        return aimePlayerGrid, action


class CheckBoundary():
    def __init__(self, xBoundary, yBoundary):
        self.xMin, self.xMax = xBoundary
        self.yMin, self.yMax = yBoundary

    def __call__(self, position):
        adjustedX, adjustedY = position
        if position[0] >= self.xMax:
            adjustedX = self.xMax
        if position[0] <= self.xMin:
            adjustedX = self.xMin
        if position[1] >= self.yMax:
            adjustedY = self.yMax
        if position[1] <= self.yMin:
            adjustedY = self.yMin
        checkedPosition = (adjustedX, adjustedY)
        return checkedPosition


class ModelController():
    def __init__(self, policy, gridSize, softmaxBeta):
        self.policy = policy
        self.gridSize = gridSize
        self.actionSpace = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        self.softmaxBeta = softmaxBeta

    def __call__(self, playerGrid, targetGrid1, targetGrid2):
        try:
            policyForCurrentStateDict = self.policy[(playerGrid, (targetGrid1, targetGrid2))]
        except KeyError as e:
            policyForCurrentStateDict = self.policy[(playerGrid, (targetGrid2, targetGrid1))]
        if self.softmaxBeta < 0:
            actionMaxList = [action for action in policyForCurrentStateDict.keys() if
                             policyForCurrentStateDict[action] == np.max(list(policyForCurrentStateDict.values()))]
            action = random.choice(actionMaxList)
        else:
            actionValues = list(policyForCurrentStateDict.values())
            actionValues = [min(100, v) for v in actionValues]
            # normedActionValues = [actionValue / sum(actionValues) for actionValue in actionValues]
            softmaxProbabilityList = calculateSoftmaxProbability(actionValues, self.softmaxBeta)
            action = list(policyForCurrentStateDict.keys())[
                list(np.random.multinomial(1, softmaxProbabilityList)).index(1)]
        aimePlayerGrid = tuple(np.add(playerGrid, action))
        pg.time.delay(0)
        return aimePlayerGrid, action


class ModelControllerOnline:
    def __init__(self, softmaxBeta):
        self.softmaxBeta = softmaxBeta

    def __call__(self, playerGrid, targetGrid1, targetGrid2, QDict):
        actionDict = QDict[(playerGrid, (targetGrid1, targetGrid2))]
        if self.softmaxBeta < 0:
            action = chooseMaxAcion(actionDict)
        else:
            action = chooseSoftMaxAction(actionDict, self.softmaxBeta)
        aimePlayerGrid = tuple(np.add(playerGrid, action))
        return aimePlayerGrid, action


def chooseMaxAcion(actionDict):
    actionMaxList = [action for action in actionDict.keys() if
                     actionDict[action] == np.max(list(actionDict.values()))]
    action = random.choice(actionMaxList)
    return action


def chooseSoftMaxAction(actionDict, softmaxBeta):
    actionValue = list(actionDict.values())
    softmaxProbabilityList = calculateSoftmaxProbability(actionValue, softmaxBeta)
    action = list(actionDict.keys())[
        list(np.random.multinomial(1, softmaxProbabilityList)).index(1)]
    return action

    # diff = abs(posteriorList[0] - posteriorList[1])
    # if diff < self.commitBeta / 100:
    #     posteriorList = calculateSoftmaxProbability(posteriorList, 1 / (self.commitBeta))
    # else:
    #     posteriorList = calculateSoftmaxProbability(posteriorList, self.commitBeta)

# def sigmoidScale(x, commitBeta):
#     aNew = (1 / (1 + 1 * np.exp(- 75 * commitBeta * (x - (commitBeta + 1) / 2))) + 1) / 2
#     return aNew


def commitSigmoid(x, commitBeta):  # [10,20]
    return 1 / (1 + np.exp(- commitBeta * (x - 0.5)))


def sigmoidScale(x, commitBeta):
    aNew = (1 / (1 + 1 * np.exp(- commitBeta * (x - 0.75))) + 1) / 2
    return aNew


def goalCommit(intention, commitBeta):
    a, b = intention
    if a > 0.5:
        aNew = commitSigmoid(a, commitBeta)
        bNew = 1 - aNew
    else:
        bNew = commitSigmoid(b, commitBeta)
        aNew = 1 - bNew
    commitedIntention = [aNew, bNew]
    return commitedIntention


class InferGoalPosterior:
    def __init__(self, goalPolicy, commitBeta):
        self.goalPolicy = goalPolicy
        self.commitBeta = commitBeta

    def __call__(self, playerGrid, action, target1, target2, priorList):
        targets = list([target1, target2])
        # priorList = goalCommit(priorList, self.commitBeta)

        likelihoodList = [self.goalPolicy(playerGrid, goal).get(action) for goal in targets]
        posteriorUnnormalized = [prior * likelihood for prior, likelihood in zip(priorList, likelihoodList)]
        evidence = sum(posteriorUnnormalized)
        posteriorList = [posterior / evidence for posterior in posteriorUnnormalized]
        return posteriorList


def calBasePolicy(posteriorList, actionProbList):
    basePolicyList = [np.multiply(goalProb, actionProb) for goalProb, actionProb in zip(posteriorList, actionProbList)]
    basePolicy = np.sum(basePolicyList, axis=0)
    return basePolicy


def sampleAction(actionDict):
    actionProbs = list(actionDict.values())
    action = list(actionDict.keys())[
        list(np.random.multinomial(1, actionProbs)).index(1)]
    return action


class ModelControllerWithGoal:
    def __init__(self, gridSize, softmaxBeta, goalPolicy, Q_dict, commitBeta):
        self.gridSize = gridSize
        self.actionSpace = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        self.softmaxBeta = softmaxBeta
        self.goalPolicy = goalPolicy
        self.Q_dict = Q_dict
        self.commitBeta = commitBeta

    def __call__(self, playerGrid, targetGrid1, targetGrid2, priorList):
        targets = list([targetGrid1, targetGrid2])
        actionProbList = [list(self.goalPolicy(playerGrid, goal).values()) for goal in targets]
        actionKeys = self.Q_dict[playerGrid, targetGrid1].keys()

        priorList = goalCommit(priorList, self.commitBeta)
        # actionProbs = calBasePolicy(priorList, actionProbList)
        # actionDict = dict(zip(actionKeys, actionProbs))

        # goal = list(np.random.multinomial(1, priorList)).index(1)
        # actionProb = actionProbList[goal]
        # actionDict = dict(zip(actionKeys, actionProb))

        if np.max(priorList) > 0.55:
            goal = targets[np.argmax(priorList)]
            actionProb = actionProbList[np.argmax(priorList)]
            actionDict = dict(zip(actionKeys, actionProb))
        else:
            goal = targets[np.argmin(priorList)]
            actionProb = actionProbList[np.argmin(priorList)]
            actionDict = dict(zip(actionKeys, actionProb))

        if self.softmaxBeta < 0:
            action = chooseMaxAcion(actionDict)
        else:
            action = sampleAction(actionDict)

        aimePlayerGrid = tuple(np.add(playerGrid, action))
        return aimePlayerGrid, action
