import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..')))
import glob
DIRNAME = os.path.dirname(__file__)
import matplotlib.pyplot as plt
# plt.style.use('ggplot')
import numpy as np
import pickle
from scipy.stats import ttest_ind, entropy, mannwhitneyu, ranksums
from scipy.interpolate import interp1d
from dataAnalysis import calculateSE, calculateAvoidCommitmnetZone
# from machinePolicy.onlineVIWithObstacle import RunVI
from dataAnalysis import *


class SoftmaxPolicy:
    def __init__(self, softmaxBeta):
        self.softmaxBeta = softmaxBeta

    def __call__(self, QDict, playerGrid, targetGrid, obstacles):
        actionDict = QDict[(playerGrid, targetGrid)]
        actionValues = list(actionDict.values())
        softmaxProbabilityList = calculateSoftmaxProbability(actionValues, self.softmaxBeta)
        softMaxActionDict = dict(zip(actionDict.keys(), softmaxProbabilityList))
        return softMaxActionDict


class BasePolicy:
    def __init__(self, Q_dict, softmaxBeta):
        self.Q_dict = Q_dict
        self.softmaxBeta = softmaxBeta

    def __call__(self, playerGrid, target1, target2):
        actionDict = self.Q_dict[(playerGrid, tuple(sorted((target1, target2))))]
        actionValues = list(actionDict.values())
        softmaxProbabilityList = calculateSoftmaxProbability(actionValues, self.softmaxBeta)
        softMaxActionDict = dict(zip(actionDict.keys(), softmaxProbabilityList))
        return softMaxActionDict


def calInformationGain(baseProb, conditionProb):

    return entropy(baseProb) - entropy(conditionProb)


# class CalculateActionInformation:
#     def __init__(self, initPrior, goalPolicy, basePolicy):
#         self.initPrior = initPrior
#         self.goalPolicy = goalPolicy
#         self.basePolicy = basePolicy

#     def __call__(self, trajectory, aimAction, target1, target2):
#         trajectory = list(map(tuple, trajectory))
#         targets = list([target1, target2])
#         expectedInfoList = []
#         cumulatedInfoList = []
#         priorList = self.initPrior
#         for index, (playerGrid, action) in enumerate(zip(trajectory, aimAction)):
#             likelihoodList = [self.goalPolicy(playerGrid, goal).get(action) for goal in targets]
#             posteriorUnnormalized = [prior * likelihood for prior, likelihood in zip(priorList, likelihoodList)]
#             evidence = sum(posteriorUnnormalized)

#             posteriorList = [posterior / evidence for posterior in posteriorUnnormalized]
#             prior = posteriorList

#             goalProbList = [list(self.goalPolicy(playerGrid, goal).values()) for goal in targets]
#             baseProb = list(self.basePolicy(playerGrid, target1, target2).values())

#             # expectedInfo = sum([goalPosterior * KL(goalProb, baseProb) for goalPosterior, goalProb in zip(posteriorList, goalProbList)])
#             expectedInfo = sum([goalPosterior * calInformationGain(baseProb, goalProb) for goalPosterior, goalProb in zip(posteriorList, goalProbList)])
#             expectedInfoList.append(expectedInfo)
#             cumulatedInfo = sum(expectedInfoList)
#             cumulatedInfoList.append(cumulatedInfo)

#         return cumulatedInfoList


class CalculateActionInformation:
    def __init__(self, initPrior, goalPolicy, basePolicy):
        self.initPrior = initPrior
        self.goalPolicy = goalPolicy
        self.basePolicy = basePolicy

    def __call__(self, trajectory, aimAction, target1, target2):
        trajectory = list(map(tuple, trajectory))
        goalPosteriorList = []
        priorGoal = initPrior[0]

        goal = trajectory[-1]
        targets = list([target1, target2])
        noGoal = [target for target in targets if target != goal][0]
        expectedInfoList = []
        cumulatedInfoList = []
        for playerGrid, action in zip(trajectory, aimAction):
            likelihoodGoal = self.goalPolicy(playerGrid, goal).get(action)
            likelihoodNogoal = self.goalPolicy(playerGrid, noGoal).get(action)
            posteriorGoal = (priorGoal * likelihoodGoal) / ((priorGoal * likelihoodGoal) + (1 - priorGoal) * likelihoodNogoal)
            priorGoal = posteriorGoal

            goalProb = list(self.goalPolicy(playerGrid, goal).values())
            baseProb = list(self.basePolicy(playerGrid, target1, target2).values())

            # expectedInfo = posteriorGoal * KL(goalProb, baseProb)
            expectedInfo = posteriorGoal * calInformationGain(baseProb, goalProb)
            expectedInfoList.append(expectedInfo)
            cumulatedInfo = sum(expectedInfoList)
            cumulatedInfoList.append(cumulatedInfo)

        return cumulatedInfoList


class GoalInfernce:
    def __init__(self, initPrior, goalPolicy, runVI):
        self.initPrior = initPrior
        self.goalPolicy = goalPolicy
        self.runVI = runVI

    def __call__(self, trajectory, aimAction, target1, target2, obstacles):
        trajectory = list(map(tuple, trajectory))
        goalPosteriorList = []
        priorGoal = initPrior[0]

        goal = trajectory[-1]
        targets = list([target1, target2])
        noGoal = [target for target in targets if target != goal][0]

        QDictGoal = self.runVI(goal, obstacles)
        QDictNoGoal = self.runVI(noGoal, obstacles)
        for playerGrid, action in zip(trajectory, aimAction):
            likelihoodGoal = self.goalPolicy(QDictGoal, playerGrid, goal, obstacles).get(action)
            likelihoodB = self.goalPolicy(QDictNoGoal, playerGrid, noGoal, obstacles).get(action)
            posteriorGoal = (priorGoal * likelihoodGoal) / ((priorGoal * likelihoodGoal) + (1 - priorGoal) * likelihoodB)
            goalPosteriorList.append(posteriorGoal)
            priorGoal = posteriorGoal
        goalPosteriorList.insert(0, initPrior[0])
        return goalPosteriorList


def getSoftmaxGoalPolicy(Q_dict, playerGrid, target, softmaxBeta):
    actionDict = Q_dict[(playerGrid, target)]
    actionValues = list(actionDict.values())
    softmaxProbabilityList = calculateSoftmaxProbability(actionValues, softmaxBeta)
    softMaxActionDict = dict(zip(actionDict.keys(), softmaxProbabilityList))
    return softMaxActionDict


class InferPosterior:
    def __init__(self, softmaxBeta, runVI):
        self.softmaxBeta = softmaxBeta
        self.runVI = runVI
        self.initPrior = [0.5, 0.5]

    def __call__(self, trajectory, aimAction, target1, target2, obstacles):
        trajectory = list(map(tuple, trajectory))
        priorList = self.initPrior
        posteriorsList = [priorList]
        _, _, transitionTableA, rewardA, _, V_goalA, Q_dictA, _ = self.runVI(target1, obstacles)
        _, _, transitionTableB, rewardB, _, V_goalB, Q_dictB, _ = self.runVI(target2, obstacles)
        goalQDicts = [Q_dictA, Q_dictB]
        targets = [target1, target2]

        for playerGrid, action in zip(trajectory, aimAction):
            goalPolicies = [getSoftmaxGoalPolicy(Q_dict, playerGrid, goal, self.softmaxBeta) for Q_dict, goal in zip(goalQDicts, targets)]
            likelihoodList = [goalPolicies[goalIndex].get(action) for goalIndex, goal in enumerate(targets)]
            posteriorUnnormalized = [prior * likelihood for prior, likelihood in zip(priorList, likelihoodList)]
            evidence = sum(posteriorUnnormalized)
            posteriors = [posterior / evidence for posterior in posteriorUnnormalized]
            posteriorsList.append(posteriors)
            priorList = posteriors

        return posteriorsList


def calPosteriorByInterpolation(goalPosteriorList, xInterpolation):
    x = np.divide(np.arange(len(goalPosteriorList) + 1), len(goalPosteriorList))
    goalPosteriorList.append(1)
    y = np.array(goalPosteriorList)
    f = interp1d(x, y, kind='nearest')
    goalPosterior = f(xInterpolation)
    return goalPosterior


def calPosteriorByChosenSteps(goalPosteriorList, xnew):
    goalPosterior = np.array(goalPosteriorList)[xnew]
    return goalPosterior


def calInfo(expectedInfoList):
    x = np.divide(np.arange(len(expectedInfoList)), len(expectedInfoList) - 1)
    y = np.array(expectedInfoList)
    f = interp1d(x, y, kind='nearest')
    xnew = np.linspace(0., 1., 30)
    goalPosterior = f(xnew)
    return goalPosterior


class CalFirstIntentionStep:
    def __init__(self, inferThreshold):
        self.inferThreshold = inferThreshold

    def __call__(self, goalPosteriorList):
        for index, goalPosteriori in enumerate(goalPosteriorList):
            if goalPosteriori > self.inferThreshold:
                return index + 1
                break
        return len(goalPosteriorList)


class CalFirstIntentionStepRatio:
    def __init__(self, calFirstIntentionStep):
        self.calFirstIntentionStep = calFirstIntentionStep

    def __call__(self, goalPosteriorList):
        firstIntentionStep = self.calFirstIntentionStep(goalPosteriorList)
        firstIntentionStepRatio = firstIntentionStep / len(goalPosteriorList)
        return firstIntentionStepRatio


def isDecisionStepInZone(trajectory, target1, target2, decisionSteps):
    trajectory = list(map(tuple, trajectory))[:decisionSteps + 1]
    initPlayerGrid = trajectory[0]
    zone = calculateAvoidCommitmnetZone(initPlayerGrid, target1, target2)
    isStepInZone = [step in zone for step in trajectory[1:]]
    return np.all(isStepInZone)


def calGoalPosteriorFromAll(posteriors, trajectory, target1, target2):
    trajectory = list(map(tuple, trajectory))
    goalIndex = None
    if trajectory[-1] == target1:
        goalIndex = 0
    elif trajectory[-1] == target2:
        goalIndex = 1
    else:
        print("trajectory no goal reach! ")
        print(trajectory, target1, target2)
    goalPosteriorList = [posterior[goalIndex] for posterior in posteriors]
    return goalPosteriorList


if __name__ == '__main__':
    from machinePolicy.showIntentionModel import RunVI, GetShowIntentionPolices

    gridSize = 15
    noise = 0.067
    gamma = 0.9
    goalReward = 30
    actionSpace = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    noiseActionSpace = [(0, -1), (0, 1), (-1, 0), (1, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)]
    runVI = RunVI(gridSize, actionSpace, noiseActionSpace, noise, gamma, goalReward)

    softmaxBeta = 2.5
    inferPosterior = InferPosterior(softmaxBeta, runVI)

    resultsPath = os.path.join(os.path.join(DIRNAME, '..'), 'results')

    participants = ['human', 'RL']
    # participants = ['human', 'intentionModel/threshold0.5infoScale11']
    # participants = ['intentionModel/threshold0.3infoScale11', 'intentionModel/threshold0.3infoScale8']
    # participants = ['human', 'intentionModelChosen/threshold0.07infoScale8.5']
    # participants = ['human', 'intentionModel/threshold0.1infoScale7softmaxBetaInfer3']
    # participants = ['intentionModelChosen/threshold0.07infoScale8.5', 'intentionModel/threshold0.07infoScale8.5']
    # participants = ['human']
    # participants = ['human', 'intentionModel/threshold0.1infoScale7softmaxBetaInfer3', 'RL']
    # participants = ['human']
    # participants = ['test/human', 'test/intention', 'test/RL']

    # decisionStep = 2
    for decisionStep in [6]:  # , 4, 2, 1, 0]:
        statsList = []
        semList = []
        statDFList = []

        for participant in participants:
            dataPath = os.path.join(resultsPath, participant)
            df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False)
            nubOfSubj = len(df["name"].unique())
            print(participant, nubOfSubj)

            df = df[(df['targetDiff'] == '0') | (df['targetDiff'] == 0)]
            df = df[(df['conditionName'] == 'expCondition1') | (df['conditionName'] == 'expCondition2')]
            df = df[(df['decisionSteps'] == decisionStep)]

            df["trajLength"] = df.apply(lambda x: len(eval(x['trajectory'])), axis=1)

            df['isValidTraj'] = df.apply(lambda x: isValidTraj(eval(x['trajectory']), eval(x['target1']), eval(x['target2'])), axis=1)
            df = df[df['isValidTraj'] == 1]

            chosenSteps = 16
            df = df[(df["trajLength"] > chosenSteps)]

            # df = df[(df["trajLength"] > 14) & (df["trajLength"] < 25)]
            # df['goalPosterior'] = df.apply(lambda x: goalInfernce(eval(x['trajectory']), eval(x['aimAction']), eval(x['target1']), eval(x['target2']), eval(x['obstacles'])), axis=1)

            # df['goalPosteriorList'] = df.apply(lambda x: goalInfernce(eval(x['trajectory']), eval(x['aimAction']), eval(x['target1']), eval(x['target2']), eval(x['obstacles'])), axis=1)

            df['posteriors'] = df.apply(lambda x: inferPosterior(eval(x['trajectory']), eval(x['aimAction']), eval(x['target1']), eval(x['target2']), eval(x['obstacles'])), axis=1)
            # df.to_csv("humanPosterior.csv")

            df['goalPosteriorList'] = df.apply(lambda x: calGoalPosteriorFromAll(x['posteriors'], eval(x['trajectory']), eval(x['target1']), eval(x['target2'])), axis=1)

    # interpolation
            # xnew = np.linspace(0., 1., 15)
            # df['goalPosterior'] = df.apply(lambda x: calPosteriorByInterpolation(x['goalPosteriorList'], xnew), axis=1)

            xnew = np.array(list(range(chosenSteps + 1)))
            df['goalPosterior'] = df.apply(lambda x: calPosteriorByChosenSteps(x['goalPosteriorList'], xnew), axis=1)
            # df['goalPosterior'] = df.apply(lambda x: np.round(np.array(x['goalPosterior']) * 100), axis=1)

            def arrMean(df, colnames):
                arr = np.array(df[colnames].tolist())
                return np.mean(arr, axis=0)
            statDF = pd.DataFrame()
            statDF['goalPosterior'] = df.groupby(['name']).apply(arrMean, 'goalPosterior')

            df = statDF
            goalPosterior = np.array(df['goalPosterior'].tolist())
            goalPosteriorMean = np.mean(goalPosterior, axis=0)

            degreeOfFreedom = len(goalPosterior) - 1  # 20-1

            goalPosteriorStd = np.divide(np.std(goalPosterior, axis=0, ddof=1), np.sqrt(len(goalPosterior)))
            statsList.append(goalPosteriorMean)
            semList.append(goalPosteriorStd)

            grouped = pd.DataFrame(df.groupby('name').apply(arrMean, 'goalPosterior'))
            statArr = np.round(np.array(grouped.iloc[:, 0].tolist()).T, 1)
            statDFList.append(statArr)

    #     pvalus = np.array([ttest_ind(statDFList[0][i], statDFList[1][i])[1] for i in range(statDFList[0].shape[0])])

    # print(statDFList)
    # statDF['goalPosterior'] = df.groupby(['name', 'beanEaten']).apply(arrMean, 'goalPosterior')
    # statDF = statDF.reset_index()
    # eatOld = statDF[statDF['beanEaten'] == 1]
    # eatNew = statDF[statDF['beanEaten'] == 2]

    # condition1 = np.array(eatOld['goalPosterior'].tolist())
    # condition2 = np.array(eatNew['goalPosterior'].tolist())

    # def arrMean(df, colnames):
    #     arr = np.array(df[colnames].tolist())
    #     return np.mean(arr, axis=0)

    print(statDFList[0].shape)

    condition1 = statDFList[0].tolist()
    condition2 = statDFList[1].tolist()
    # print(condition1.shape)
    # statDF['goalPosterior'] = df.groupby(['name'])["goalPosterior"].mean().sort_index()

    import mne
    from mne import io
    from mne.stats import permutation_cluster_test
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test([condition1, condition2], n_permutations=1000,
                                                                     tail=1,
                                                                     out_type='mask')
    print(T_obs)
    print(clusters)
    print(cluster_p_values)
