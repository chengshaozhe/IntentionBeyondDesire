import pandas as pd
import os
import glob
DIRNAME = os.path.dirname(__file__)
import matplotlib.pyplot as plt
# plt.style.use('ggplot')
import numpy as np
import pickle
from scipy.stats import ttest_ind, entropy
from scipy.interpolate import interp1d
from sklearn.metrics import mutual_info_score as KL
from dataAnalysis import calculateSE, calculateSD, calculateAvoidCommitmnetZoneAll, calculateAvoidCommitmnetZone, calMidPoints, calculateFirstIntentionConsistency
import researchpy
import seaborn as sns
from scipy.stats import t, norm


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


def calParticipantType(name):
    if 'max' in name:
        participantsType = 'Desire Model'
    if 'intention' in name:
        participantsType = 'Intention Model'
    else:
        participantsType = 'Humans'

    return participantsType


def isValidTraj(trajectory, target1, target2):
    trajectory = list(map(tuple, trajectory))
    finalState = trajectory[-1]
    if finalState == target1 or finalState == target2:
        return True
    else:
        return False


if __name__ == '__main__':
    resultsPath = os.path.join(os.path.join(DIRNAME, '..'), 'results')
    # participants = ['noise0.0673_softmaxBeta2.5']
    # participants = ['human0331', 'RL']
    # participants = ['human', 'intentionModelWithSpMonitor/threshold0infoScale5.5']
    # participants = ['human', 'intentionModelWithNaiveInfer/threshold0.5infoScale11']
    # participants = ['human', 'intentionModel3/threshold0.5infoScale3']
    # participants = ['human']
    participants = ['human', 'RL']
    # participants = ['human', 'intentionModelChosen/threshold0.01infoScale8.5']

    # participants = ['human', 'intentionModel/threshold0.08infoScale8.5']

    dataPaths = [os.path.join(resultsPath, participant) for participant in participants]
    dfList = [pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False) for dataPath in dataPaths]
    df = pd.concat(dfList, sort=True)

    # first half
    # df = df[df.index < 144]

    df['targetDiff'] = df.apply(lambda x: str(x['targetDiff']), axis=1)
    # df = df[(df['targetDiff'] == '0')]  # main result

    # df = df[(df['targetDiff'] != '0')]

    # df['participantsType'] = ['MEU' if 'max' in name else 'Human' for name in df['name']]
    df['participantsType'] = df.apply(lambda x: calParticipantType(x['name']), axis=1)
    df['totalStep'] = df.apply(lambda x: len(eval(x['trajectory'])), axis=1)

    # df['isValidTraj'] = df.apply(lambda x: isValidTraj(eval(x['trajectory']), eval(x['target1']), eval(x['target2'])), axis=1)
    # df = df[df['isValidTraj'] == 1]

    # print(df.groupby(['participantsType'])['totalStep'].agg(['mean', 'min', 'max', 'count']))

    # df = df[(df["totalStep"] < 32)]

    # df['totalTime'] = df.apply(lambda x: eval(x['reactionTime'])[-1], axis=1)

    # df = df[(df['noisePoint'] == '[]')]

    # df = df[(df['targetDiff'] == 0) & (df['isDecisionStepInZone'] == 1)]

    # df = df[(df['targetDiff'] == '0') | (df['targetDiff'] == 0)]

    dfExpTrail = df[(df['conditionName'] == 'expCondition1') | (df['conditionName'] == 'expCondition2')]
    # dfExpTrail = df[(df['conditionName'] == 'expCondition1')]

    dfExpTrail['hasAvoidPoint'] = dfExpTrail.apply(lambda x: hasAvoidPoints(eval(x['aimPlayerGridList']), eval(x['avoidCommitPoint'])), axis=1)

    # dfExpTrail.to_csv('dfExpTrail.csv')
    statDF = pd.DataFrame()
    statDF['avoidCommitPercent'] = dfExpTrail.groupby(['name', 'decisionSteps'])["hasAvoidPoint"].mean()
    # statDF['ShowCommitmentPercent'] = statDF.apply(lambda x: (1 - x['avoidCommitPercent']), axis=1)

    statDF['ShowCommitmentPercent'] = statDF.apply(lambda x: int((1 - x['avoidCommitPercent']) * 100), axis=1)

    statDF = statDF.reset_index()
    # statDF['participantsType'] = ['RL' if 'noise' in name else 'Humans' for name in statDF['name']]

    # statDF['participantsType'] = statDF.apply(lambda x: calParticipantType(x['name']), axis=1)
    statDF['participantsType'] = ['MEU' if 'noise' in name else 'Human' for name in statDF['name']]

    # statDF['avoidCommitPercentSE'] = statDF["avoidCommitPercent"].apply(calculateSE)

    # statDF['meanReactionTime'] = [meanTime[name] for name in statDF['name']]

    # statDF['sem'] = df.groupby(['participantsType', 'decisionSteps'])["avoidCommitPercent"].apply(calculateSE)

    # statDF = statDF[statDF['participantsType'] == 'Human']
    # statDF = statDF[statDF['participantsType'] == 'RL Agent']

    # print(statDF)
    # statDF.to_csv('allhumansStats.csv')

    sns.set_theme(style="white")
    plt.rcParams['figure.figsize'] = (8, 6)
    # plt.rcParams['figure.dpi'] = 200

    colorList = [(0.8392156862745098, 0.15294117647058825, 0.1568627450980392),  # red
                 (0.12156862745098039, 0.4666666666666667, 0.7058823529411765)]  # blue
    ax = sns.barplot(x="decisionSteps", y="ShowCommitmentPercent", hue="participantsType", data=statDF, ci=None, palette=colorList)
    # ax = sns.boxplot(x="decisionSteps", y="ShowCommitmentPercent", hue="participantsType", data=statDF, palette="Set1", showmeans=True)

    plt.axhline(y=50, color='k', linestyle='--', alpha=0.5)

    def changeRectWidth(ax, new_value):
        xList = []
        yList = []
        for index, bar in enumerate(ax.patches):
            current_width = bar.get_width()
            diff = current_width - new_value
            bar.set_width(new_value)
            if index < len(ax.patches) / 2:
                bar.set_x(bar.get_x() + diff)
            xList.append(bar.get_x() + diff / 2.)
            yList.append(bar.get_height())
        return xList, yList

    xList, yList = changeRectWidth(ax, 0.2)

    stats = statDF.groupby(['decisionSteps', 'participantsType'], sort=False)['ShowCommitmentPercent'].agg(['mean', 'count', 'std'])
    # print(stats)
    # print('-' * 50)

    ci_lo = []
    ci_hi = []

    for i in stats.index:
        m, c, s = stats.loc[i]
        alpha = 0.05
        t_ci = t.ppf(1 - alpha / 2, 19)  # two-tailed 95% , N=20 t_ci = 2.093

        ci_lo.append(m - t_ci * s / np.sqrt(c))
        ci_hi.append(m + t_ci * s / np.sqrt(c))

    stats['ci_lo'] = ci_lo
    stats['ci_hi'] = ci_hi

    pd.set_option('display.max_columns', None)
    print(stats)
    print(df.groupby('participantsType')["totalStep"].mean())

    yerrList = [stats['mean'] - stats['ci_lo'], stats['ci_hi'] - stats['mean']]
    plt.errorbar(x=xList, y=yList, yerr=yerrList, fmt='none', c='k', elinewidth=2, capsize=5)

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # ax = sns.boxplot(x="decisionSteps", y="avoidCommitPercent", hue="participantsType", data=statDF, palette="Set1", showmeans=True)
    # ax.set(xlabel='Steps-to-crossroad Condition', ylabel='Show Commitment Ratio', title='Commitment with Deliberation')
    # handles, labels = ax.get_legend_handles_labels()

    plt.xticks(fontsize=14, color='black')
    plt.yticks(np.arange(0, 101, 10), fontsize=14, color='black')

    plt.xlabel('Steps-to-crossroad', fontsize=16, color='black')
    plt.ylabel("% Choosing the fixed-future path", fontsize=20, color='black')

    plt.legend(loc='best', fontsize=16)
    plt.ylim((0, 101))
    plt.rcParams['svg.fonttype'] = 'none'
    # plt.savefig('/Users/chengshaozhe/Downloads/exp2All.svg', dpi=600, format='svg')

    # plt.show()
