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
import seaborn as sns

from dataAnalysis import calculateSE, calculateAvoidCommitmnetZone
# from machinePolicy.onlineVIWithObstacle import RunVI
from dataAnalysis import *

from machinePolicy.showIntentionModel import RunVI, GetShowIntentionPolices


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


def arrMean(df, colnames):
    arr = np.array(df[colnames].tolist())
    return np.mean(arr, axis=0)


if __name__ == '__main__':
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
    # participants = ['test/human', 'test/RL']

    participants = ['all']

    dataPaths = [os.path.join(resultsPath, participant) for participant in participants]
    dfList = [pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False) for dataPath in dataPaths]
    df = pd.concat(dfList, sort=True)

    df['participantsType'] = ['MEU' if 'noise' in name else 'Human' for name in df['name']]
    df['targetDiff'] = df.apply(lambda x: str(x['targetDiff']), axis=1)
    df["trajLength"] = df.apply(lambda x: len(eval(x['trajectory'])), axis=1)

    df = df[(df['targetDiff'] == '0')]
    df = df[(df['conditionName'] == 'expCondition1') | (df['conditionName'] == 'expCondition2')]
    df = df[(df['decisionSteps'] == 6)]

    # df['isValidTraj'] = df.apply(lambda x: isValidTraj(eval(x['trajectory']), eval(x['target1']), eval(x['target2'])), axis=1)
    # df = df[df['isValidTraj'] == 1]

    chosenSteps = 16
    df = df[(df["trajLength"] > chosenSteps)]

    # df['posteriors'] = df.apply(lambda x: inferPosterior(eval(x['trajectory']), eval(x['aimAction']), eval(x['target1']), eval(x['target2']), eval(x['obstacles'])), axis=1)
    # df['goalPosteriorList'] = df.apply(lambda x: calGoalPosteriorFromAll(x['posteriors'], eval(x['trajectory']), eval(x['target1']), eval(x['target2'])), axis=1)
    # df.to_csv('dataWithPosterior.csv')

    df['goalPosteriorList'] = df.apply(lambda x: eval(x['goalPosteriorList']), axis=1)
    xnew = np.array(list(range(chosenSteps + 1)))
    df['goalPosterior'] = df.apply(lambda x: calPosteriorByChosenSteps(x['goalPosteriorList'], xnew), axis=1)

    statDF = pd.DataFrame()
    statDF['goalPosterior'] = df.groupby(['name', 'participantsType']).apply(arrMean, 'goalPosterior')
    statDF = statDF.reset_index()

    # humanPosterior = np.round(np.array(grouped1.iloc[:, 0].tolist()).T, 1)
    # modelPosterior = np.round(np.array(grouped2.iloc[:, 0].tolist()).T, 1)

    humanDf = statDF[statDF.participantsType == 'Human']
    modelDf = statDF[statDF.participantsType == 'MEU']

    humanPosterior = np.round(np.array(humanDf['goalPosterior'].tolist()), 2)
    modelPosterior = np.round(np.array(modelDf['goalPosterior'].tolist()), 2)

    import mne
    from mne.stats import permutation_cluster_test
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test([humanPosterior, modelPosterior], n_permutations=1000, tail=1, out_type='mask')
    print(T_obs)
    print(clusters)
    print(cluster_p_values)

    statsList = []
    stdList = []
    statDFList = []
    for goalPosteriorArrMean in [humanPosterior, modelPosterior]:
        goalPosteriorMean = np.mean(goalPosteriorArrMean, axis=0)
        goalPosteriorSem = np.divide(np.std(goalPosteriorArrMean, axis=0, ddof=1), np.sqrt(len(goalPosteriorArrMean)))

        statsList.append(goalPosteriorMean)
        stdList.append(goalPosteriorSem)
        statDFList.append(goalPosteriorArrMean.T)
        degreeOfFreedom = len(goalPosteriorArrMean) - 1  # 20-1

    pvalus = np.array([ttest_ind(statDFList[0][i], statDFList[1][i])[1] for i in range(statDFList[0].shape[0])])
    sigArea = np.where(pvalus < 0.001)[0]
    print(sigArea)

    sns.set_theme(style="white")
    colorList = [(0.8392156862745098, 0.15294117647058825, 0.1568627450980392),  # red
                 (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),  # blue
                 (0.6, 0.6, 0.6)]  # grey

    fig, ax = plt.subplots()
    plt.rcParams['figure.dpi'] = 200
    lables = ['Humans', 'MEU']

    from scipy.stats import t, norm
    alpha = 0.05
    print(degreeOfFreedom)
    t_ci = t.ppf(1 - alpha / 2, degreeOfFreedom)  # t(19)=2.093
    for i in range(len(statsList)):
        ci95 = t_ci * stdList[i]
        plt.plot(xnew, statsList[i], label=lables[i], color=colorList[i])
        plt.fill(np.concatenate([xnew, xnew[::-1]]), np.concatenate([statsList[i] - ci95, (statsList[i] + ci95)[::-1]]), alpha=.3, color=colorList[i])

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.tick_params(top=False, bottom=False, left=True, right=False)

    plt.legend(loc='lower right', fontsize=16)
    plt.legend(loc='best', fontsize=12)

    plt.xlabel("Agent's steps over time", fontsize=14, color='black')
    plt.ylabel('Posterior probability of the actual destination', fontsize=14, color='black')
    plt.ylim((0.47, 1.05))

    plt.xticks(fontsize=12, color='black')
    plt.yticks(fontsize=12, color='black')
    plt.rcParams['svg.fonttype'] = 'none'
    # plt.show()

###
    # decisionStep = 2
    # for decisionStep in [6]:  # , 4, 2, 1, 0]:
    #     statsList = []
    #     semList = []
    #     statDFList = []
    #     dfList = []
    #     for participant in participants:
    #         dataPath = os.path.join(resultsPath, participant)
    #         df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False)
    #         nubOfSubj = len(df["name"].unique())
    #         print(participant, nubOfSubj)

    #         df = df[(df['targetDiff'] == '0') | (df['targetDiff'] == 0)]
    #         df = df[(df['conditionName'] == 'expCondition1') | (df['conditionName'] == 'expCondition2')]
    #         df = df[(df['decisionSteps'] == decisionStep)]

    #         df["trajLength"] = df.apply(lambda x: len(eval(x['trajectory'])), axis=1)

    #         df['isValidTraj'] = df.apply(lambda x: isValidTraj(eval(x['trajectory']), eval(x['target1']), eval(x['target2'])), axis=1)
    #         df = df[df['isValidTraj'] == 1]

    #         chosenSteps = 16
    #         df = df[(df["trajLength"] > chosenSteps)]

    #         df['posteriors'] = df.apply(lambda x: inferPosterior(eval(x['trajectory']), eval(x['aimAction']), eval(x['target1']), eval(x['target2']), eval(x['obstacles'])), axis=1)

    #         df['goalPosteriorList'] = df.apply(lambda x: calGoalPosteriorFromAll(x['posteriors'], eval(x['trajectory']), eval(x['target1']), eval(x['target2'])), axis=1)

    # # interpolation
    #         # xnew = np.linspace(0., 1., 15)
    #         # df['goalPosterior'] = df.apply(lambda x: calPosteriorByInterpolation(x['goalPosteriorList'], xnew), axis=1)

    #         xnew = np.array(list(range(chosenSteps + 1)))
    #         df['goalPosterior'] = df.apply(lambda x: calPosteriorByChosenSteps(x['goalPosteriorList'], xnew), axis=1)
    #         # df['goalPosterior'] = df.apply(lambda x: np.round(np.array(x['goalPosterior']) * 100), axis=1)

    #         statDF = pd.DataFrame()
    #         statDF['goalPosterior'] = df.groupby(['name']).apply(arrMean, 'goalPosterior')

    #         df = statDF
    #         goalPosterior = np.array(df['goalPosterior'].tolist())
    #         goalPosteriorMean = np.mean(goalPosterior, axis=0)

    #         degreeOfFreedom = len(goalPosterior) - 1  # 20-1

    #         goalPosteriorStd = np.divide(np.std(goalPosterior, axis=0, ddof=1), np.sqrt(len(goalPosterior)))
    #         statsList.append(goalPosteriorMean)
    #         semList.append(goalPosteriorStd)

    #         grouped = pd.DataFrame(df.groupby('name').apply(arrMean, 'goalPosterior'))
    #         statArr = np.round(np.array(grouped.iloc[:, 0].tolist()).T, 1)
    #         statDFList.append(statArr)
    #         dfList.append(statDF)

    #     pvalus = np.array([ttest_ind(statDFList[0][i], statDFList[1][i])[1] for i in range(statDFList[0].shape[0])])

    #     sigArea = np.where(pvalus < 0.05)[0]
    #     print(sigArea)

    #     humanPosterior = statDFList[0]
    #     modelPosterior = statDFList[1]

    #     print(humanPosterior.shape)

    #     import mne
    #     from mne.stats import permutation_cluster_test
    #     T_obs, clusters, cluster_p_values, H0 = \
    #         permutation_cluster_test([humanPosterior, modelPosterior], n_permutations=1000, tail=1, out_type='mask')
    #     print(T_obs)
    #     print(clusters)
    #     print(cluster_p_values)

    #     # lables = ['Humans', 'Intention Model', 'RL']
    #     lables = ['Humans', 'MEU Agent']

    #     lineWidth = 1
    #     # xnew = np.array(list(range(1, 16)))
    #     fig, ax = plt.subplots()
    #     plt.rcParams['figure.dpi'] = 200

    #     colorList = [(0.8392156862745098, 0.15294117647058825, 0.1568627450980392),  # red
    #                  (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),  # blue
    #                  (0.6, 0.6, 0.6)]  # grey
    #     for i in range(len(statsList)):
    #         plt.plot(xnew, statsList[i], label=lables[i], linewidth=lineWidth, color=colorList[i])
    #         # plt.errorbar(xnew, statsList[i], yerr=ci95, label=lables[i])

    #         from scipy.stats import t, norm
    #         alpha = 0.05
    #         # ci95t = semList[i] * t.ppf(1 - alpha / 2, degreeOfFreedom)
    #         t_ci = 2.093  # two-tailed 95% z_ci = 1.96
    #         ci95 = t_ci * semList[i]
    #         plt.fill(np.concatenate([xnew, xnew[::-1]]), np.concatenate([statsList[i] - ci95, (statsList[i] + ci95)[::-1]]), color=colorList[i], alpha=.2)

    #     # sns.regplot(xnew, statsList[i], data=ans.loc[ans.dataset == "III"], scatter_kws={"s": 80},robust=True, ci=95)

    #     ax.spines['right'].set_color('none')
    #     ax.spines['top'].set_color('none')

    # # sig area line
    #     # xnewSig = xnew[sigArea]
    #     # ySig = [stats[sigArea] for stats in statsList]
    #     # for sigLine in [xnewSig[0], xnewSig[-1]]:
    #     #     plt.plot([sigLine] * 10, np.linspace(0.5, 1., 10), color='black', linewidth=2, linestyle="--")

    #     plt.legend(loc='best', fontsize=12)
    #     plt.xlabel("Agent's steps over time", fontsize=14, color='black')
    #     plt.ylabel('Posterior probability of goal-reached', fontsize=14, color='black')
    #     plt.ylim((0.47, 1.05))

    #     plt.xticks(fontsize=12, color='black')
    #     plt.yticks(fontsize=12, color='black')
    #     plt.rcParams['svg.fonttype'] = 'none'
    #     # plt.savefig('/Users/chengshaozhe/Downloads/exp2bStep{}.svg'.format(str(decisionStep)), dpi=600, format='svg')

    #     # plt.title('Inferred Goal Through Time', fontsize=fontSize, color='black')
    #     plt.show()
