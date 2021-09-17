import numpy as np
from scipy.interpolate import interp1d


def calculateSE(data):
    standardError = np.std(data, ddof=1) / np.sqrt(len(data))
    return standardError


def calculateGridDis(grid1, grid2):
    gridDis = np.linalg.norm(np.array(grid1) - np.array(grid2), ord=1)
    return gridDis


def creatRect(coor1, coor2):
    vector = np.array(list(zip(coor1, coor2)))
    vector.sort(axis=1)
    rect = [(i, j) for i in range(vector[0][0], vector[0][1] + 1) for j in range(vector[1][0], vector[1][1] + 1)]
    return rect


def calculateAvoidCommitmnetZone(playerGrid, target1, target2):
    dis1 = calculateGridDis(playerGrid, target1)
    dis2 = calculateGridDis(playerGrid, target2)
    if dis1 == dis2:
        rect1 = creatRect(playerGrid, target1)
        rect2 = creatRect(playerGrid, target2)
        avoidCommitmentZone = list(set(rect1).intersection(set(rect2)))
        avoidCommitmentZone.remove(tuple(playerGrid))
    else:
        avoidCommitmentZone = []
    return avoidCommitmentZone


def calculateAvoidCommitmentRatio(trajectory, zone):
    avoidCommitmentSteps = 0
    for step in trajectory:
        if tuple(step) in zone:
            avoidCommitmentSteps += 1
    avoidCommitmentRatio = avoidCommitmentSteps / (len(trajectory) - 1)
    return avoidCommitmentRatio


def calculateFirstOutZoneRatio(trajectory, zone):
    avoidCommitmentPath = list()
    for point in trajectory:
        if tuple(point) not in zone and len(avoidCommitmentPath) != 0:
            break
        if tuple(point) in zone:
            avoidCommitmentPath.append(point)
    avoidCommitmentRatio = len(avoidCommitmentPath) / (len(trajectory) - 1)
    return avoidCommitmentRatio


def calculateFirstIntentionStep(goalList):
    goal1Step = goal2Step = len(goalList)
    if 1 in goalList:
        goal1Step = goalList.index(1) + 1
    if 2 in goalList:
        goal2Step = goalList.index(2) + 1
    firstIntentionStep = min(goal1Step, goal2Step)
    return firstIntentionStep


def calculateFirstIntentionRatio(goalList):
    firstIntentionStep = calculateFirstIntentionStep(goalList)
    firstIntentionRatio = firstIntentionStep / len(goalList)
    return firstIntentionRatio


def calculateFirstIntention(goalList):
    try:
        target1Goal = goalList.index(1)
    except ValueError as e:
        target1Goal = 999
    try:
        target2Goal = goalList.index(2)
    except ValueError as e:
        target2Goal = 999
    if target1Goal < target2Goal:
        firstGoal = 1
    elif target2Goal < target1Goal:
        firstGoal = 2
    else:
        firstGoal = 0
    return firstGoal


def calculateFirstIntentionConsistency(goalList):
    firstGoal = calculateFirstIntention(goalList)
    finalGoal = calculateFirstIntention(list(reversed(goalList)))
    firstIntention = 1 if firstGoal == finalGoal else 0
    return firstIntention


def judgeStraightCondition(player, target1, target2):
    if player[0] == target1[0] or player[0] == target2[0] or player[1] == target1[1] or player[1] == target2[1]:
        straightCondition = True
    else:
        straightCondition = False
    return straightCondition


def calculateSoftmaxProbability(acionValues, beta):
    newProbabilityList = list(np.divide(np.exp(np.multiply(beta, acionValues)), np.sum(np.exp(np.multiply(beta, acionValues)))))
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


class GoalInfernce:
    def __init__(self, initPrior, goalPolicy):
        self.initPrior = initPrior
        self.goalPolicy = goalPolicy

    def __call__(self, trajectory, aimAction, target1, target2):
        trajectory = list(map(tuple, trajectory))
        goalPosteriorList = []
        priorGoal = self.initPrior[0]

        goal = trajectory[-1]
        targets = list([target1, target2])
        noGoal = [target for target in targets if target != goal][0]
        for playerGrid, action in zip(trajectory, aimAction):
            likelihoodGoal = self.goalPolicy(playerGrid, goal).get(action)
            likelihoodB = self.goalPolicy(playerGrid, noGoal).get(action)
            posteriorGoal = (priorGoal * likelihoodGoal) / ((priorGoal * likelihoodGoal) + (1 - priorGoal) * likelihoodB)
            goalPosteriorList.append(posteriorGoal)
            priorGoal = posteriorGoal
        goalPosteriorList.insert(0, self.initPrior[0])
        return goalPosteriorList


def calPosterior(goalPosteriorList, xnew):
    x = np.divide(np.arange(len(goalPosteriorList) + 1), len(goalPosteriorList))
    goalPosteriorList.append(1)
    y = np.array(goalPosteriorList)
    f = interp1d(x, y, kind='nearest')
    goalPosterior = f(xnew)
    return goalPosterior


def arrMean(df, colnames):
    arr = np.array(df[colnames].tolist())
    return np.mean(arr, axis=0)
