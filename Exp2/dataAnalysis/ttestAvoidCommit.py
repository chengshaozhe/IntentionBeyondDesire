import pandas as pd
import os
import glob
DIRNAME = os.path.dirname(__file__)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import pickle
from scipy.stats import ttest_ind, entropy
from scipy.interpolate import interp1d
from sklearn.metrics import mutual_info_score as KL
from dataAnalysis import calculateSE, calculateSD, calculateAvoidCommitmnetZoneAll, calculateAvoidCommitmnetZone, calMidPoints, calculateFirstIntentionConsistency
import researchpy


def calAvoidPoints(playerGrid, decisionSteps):
    addSteps = decisionSteps / 2 + 1
    x, y = playerGrid
    if x < 7 and y < 7:
        avoidPoint = (x + addSteps, y + addSteps)
    if x < 7 and y > 7:
        avoidPoint = (x + addSteps, y - addSteps)
    if x > 7 and y < 7:
        avoidPoint = (x - addSteps, y + addSteps)
    elif x > 7 and y > 7:
        avoidPoint = (x - addSteps, y - addSteps)
    return avoidPoint


def isGridsALine(playerGrid, targetGrid):
    if playerGrid[0] == targetGrid[0] or playerGrid[1] == targetGrid[1]:
        return True
    else:
        return False


def isTrajHasAvoidPoints(trajectory, aimAction, initPlayerGrid, target1, target2, decisionSteps, conditionName, obstacles):
    trajectory = list(map(tuple, trajectory))
    if conditionName == 'expCondition':
        avoidPoint = calAvoidPoints(initPlayerGrid, decisionSteps)
        hasMidPoint = 1 if avoidPoint in trajectory else 0
        if decisionSteps == 0:
            nextStep = trajectory[1]
            nextStepInLineWithObstacles = [isGridsALine(nextStep, targetGrid) for targetGrid in obstacles]
            hasMidPoint = 1 if sum(nextStepInLineWithObstacles) > 2 else 0
        if decisionSteps == 1:
            avoidPoint = calAvoidPoints(initPlayerGrid, decisionSteps - 1)
            hasMidPoint = 1 if avoidPoint in trajectory else 0

    if conditionName == 'lineCondition':
        avoidPoint = calMidPoints(initPlayerGrid, target1, target2)
        hasMidPoint = 1 if avoidPoint in trajectory else 0
        # hasMidPoint = 1 if aimAction[decisionSteps] == aimAction[decisionSteps - 1] else 0
    return hasMidPoint


def hasAvoidPoints(trajectory, avoidPoint):
    trajectory = list(map(tuple, trajectory))
    hasMidPoint = 1 if avoidPoint in trajectory else 0
    return hasMidPoint


def sliceTraj(trajectory, midPoint):
    trajectory = list(map(tuple, trajectory))
    index = trajectory.index(midPoint) + 1
    return trajectory[:index]


def isDecisionStepInZone(trajectory, target1, target2, decisionSteps):
    trajectory = list(map(tuple, trajectory))[:decisionSteps + 1]
    initPlayerGrid = trajectory[0]
    zone = calculateAvoidCommitmnetZone(initPlayerGrid, target1, target2)
    isStepInZone = [step in zone for step in trajectory[1:]]
    return np.all(isStepInZone)


if __name__ == '__main__':
    machinePolicyPath = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), 'machinePolicy'))

    resultsPath = os.path.join(os.path.join(DIRNAME, '..'), 'results')
    participants = ['human', 'RL']
    # participants = ['human0331']
    # participants = ['intentionModelWithSophistictedInfer2/threshold0.08infoScale9']

    dataPaths = [os.path.join(resultsPath, participant) for participant in participants]
    dfList = [pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False) for dataPath in dataPaths]
    df = pd.concat(dfList, sort=True)
    df['participantsType'] = ['RL Agent' if 'max' in name else 'Human' for name in df['name']]

    #!!!!!!
    # df['name'] = df.apply(lambda x: x['name'][:-1], axis=1)

    # df['isDecisionStepInZone'] = df.apply(lambda x: isDecisionStepInZone(eval(x['trajectory']), eval(x['target1']), eval(x['target2']), x['decisionSteps']), axis=1)
    df['totalTime'] = df.apply(lambda x: eval(x['reactionTime'])[-1], axis=1)

    df['targetDiff'] = df.apply(lambda x: str(x['targetDiff']), axis=1)

    # df = df[(df['noisePoint'] == '[]')]

    # df = df[(df['targetDiff'] == 0) & (df['isDecisionStepInZone'] == 1)]

    df = df[(df['targetDiff'] == '0')]  # main result
    # df = df[df['decisionSteps'] == 0]

# first half
    # df = df[df.index < 144]

    # dfExpTrail = df[(df['conditionName'] == 'expCondition2')]

    dfExpTrail = df[(df['conditionName'] == 'expCondition1') | (df['conditionName'] == 'expCondition2')]

    # dfExpTrail['hasAvoidPoint'] = dfExpTrail.apply(lambda x: isTrajHasAvoidPoints(eval(x['trajectory']), eval(x['aimAction']), eval(x['playerGrid']), eval(x['target1']), eval(x['target2']), x['decisionSteps'], x['conditionName'], eval(x['obstacles'])), axis=1)

    dfExpTrail['hasAvoidPoint'] = dfExpTrail.apply(lambda x: hasAvoidPoints(eval(x['aimPlayerGridList']), eval(x['avoidCommitPoint'])), axis=1)

    statDF = pd.DataFrame()
    # statDF['avoidCommitPercent'] = dfExpTrail.groupby(['name', 'decisionSteps'])["hasAvoidPoint"].mean()
    statDF['avoidCommitPercent'] = dfExpTrail.groupby(['name'])["hasAvoidPoint"].mean()

    # statDF['avoidCommitPercent'] = dfExpTrail.groupby(['name'])["hasAvoidPoint"].mean()

    statDF['ShowCommitmentPercent'] = statDF.apply(lambda x: 1 - x['avoidCommitPercent'], axis=1)

    statDF = statDF.reset_index()
    statDF['participantsType'] = ['RL Agent' if 'max' in name else 'Human' for name in statDF['name']]

    # statDF['avoidCommitPercentSE'] = statDF["avoidCommitPercent"].apply(calculateSE)

    # statDF['meanReactionTime'] = [meanTime[name] for name in statDF['name']]

    # statDF['sem'] = df.groupby(['participantsType', 'decisionSteps'])["avoidCommitPercent"].apply(calculateSE)

    # statDF = statDF[statDF['participantsType'] == 'Human']
    statDF = statDF[statDF['participantsType'] == 'RL Agent']

    # statDF = statDF[statDF['decisionSteps'] == 1]
#
    # print(statDF)
    # dfExpTrail.to_csv('dfExpTrail.csv')

# Compute the two-way mixed-design ANOVA
    calAnova = 1
    if calAnova:
        import pingouin as pg
        pd.set_option('max_columns', 8)
        stats = pg.ttest(statDF['ShowCommitmentPercent'], 0.5)
        print(stats)
        print('mean:', np.mean(statDF['ShowCommitmentPercent']))
        # print(stats['p-val'])
        # print(stats['CI95%'])

        from scipy import stats
        pop_mean = 0.5
        t, p_twotail = stats.ttest_1samp(statDF['ShowCommitmentPercent'], pop_mean)
        print('t=', t, 'p=', p_twotail)

        # from scipy import stats
        # a = stats.ttest_1samp(statDF['ShowCommitmentPercent'], 0.5)
        # print(a)

    VIZ = 1
    if VIZ:
        import seaborn as sns
        ax = sns.barplot(x="participantsType", y="ShowCommitmentPercent", data=statDF, ci=95)
        # ax = sns.barplot(x="decisionSteps", y="ShowCommitmentPercent", hue="name", data=statDF, ci=68)

        # ax = sns.boxplot(x="decisionSteps", y="ShowCommitmentPercent", hue="participantsType", data=statDF, palette="Set1", showmeans=True)
        ax.set(xlabel='Decision Step', ylabel='Show Commitment Ratio', title='Commitment with Deliberation')
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(loc='best')
        plt.ylim((0, 1))
        # plt.show()
