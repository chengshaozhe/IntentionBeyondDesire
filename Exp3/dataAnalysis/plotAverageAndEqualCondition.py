import pandas as pd
import os
import glob
DIRNAME = os.path.dirname(__file__)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import sys
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..')))
from dataAnalysis import *

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 5000)


def isEatOld(beanEaten):
    if beanEaten == 1:
        return True
    else:
        return False


def cleanDataFrame(rawDataFrame):
    cleanConditionDataFrame = rawDataFrame[rawDataFrame.condition != "None"]
    cleanBeanEatenDataFrame = cleanConditionDataFrame[cleanConditionDataFrame.beanEaten != 0]
    return cleanBeanEatenDataFrame


def judgeStraightCondition(player, target1, target2):
    if player[0] == target1[0] or player[0] == target2[0] or player[1] == target1[1] or player[1] == target2[1]:
        straightCondition = True
    else:
        straightCondition = False
    return straightCondition


if __name__ == '__main__':
    resultsPath = os.path.join(os.path.join(DIRNAME, '..'), 'results')
#
    participants = ['human', 'RL']
    participants = ['all']

    dataPaths = [os.path.join(resultsPath, participant) for participant in participants]
    dfList = [pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False) for dataPath in dataPaths]
    df = pd.concat(dfList, sort=True)

    df["eatOld"] = df.apply(lambda x: isEatOld(x['beanEaten']), axis=1)
    df = df[df['condition'] != 'None']
    df.condition = df.apply(lambda x: int(x['condition']), axis=1)

    dfEqual = df[df['condition'] == 0]
    equalDF = pd.DataFrame()
    equalDF['eatOldRatio'] = dfEqual.groupby(['name', 'participantsType', 'condition'])["eatOld"].mean()
    equalDF['conditionType'] = 'Equal Distance'
    equalDF = equalDF.reset_index()

    statDF = pd.DataFrame()
    statDF['eatOldRatio'] = df.groupby(['name', 'participantsType'])["eatOld"].mean()
    statDF['conditionType'] = 'All Distances Averaged'
    statDF = statDF.reset_index()

    newDf = pd.concat([statDF, equalDF])
    newDf['eatOldRatio'] = newDf.apply(lambda x: np.array(x["eatOldRatio"]) * 100, axis=1)

# independent t-test
    import researchpy
    statsTable = researchpy.summary_cont(newDf.groupby(['participantsType', 'conditionType'])['eatOldRatio'])
    print(statsTable)

    testDF = pd.DataFrame()
    # testDF = newDf[newDf['conditionType'] == 'All Distances Averaged']
    testDF = newDf[newDf['conditionType'] == 'Equal Distance']

    humanDf = testDF[testDF['participantsType'] == 'Humans']
    modelDf = testDF[testDF['participantsType'] == 'MEU']

    x1 = humanDf.groupby('name')['eatOldRatio'].mean()
    x2 = modelDf.groupby('name')['eatOldRatio'].mean()

    des, res = researchpy.ttest(x1, x2)

    print(des)
    print(res)

    # newDf.to_csv('statDf.csv')

# plotting
    sns.set_theme(style="white")
    plt.rcParams['figure.dpi'] = 200

    colorList = [(0.8392156862745098, 0.15294117647058825, 0.1568627450980392),  # red
                 (0.12156862745098039, 0.4666666666666667, 0.7058823529411765)]
    ax = sns.barplot(x="conditionType", y="eatOldRatio", hue="participantsType", data=newDf, ci=None, palette=colorList)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

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

    stats = newDf.groupby(['participantsType', 'conditionType'])['eatOldRatio'].agg(['mean', 'count', 'std'])

    # print(stats)
    # print('-' * 50)

    sem_hi = []
    sem_lo = []
    for i in stats.index:
        m, c, s = stats.loc[i]
        t_ci = 2.093  # t(19)=2.093 two-tailed 95%
        # sem_hi.append(m + s / np.sqrt(c))
        # sem_lo.append(m - s / np.sqrt(c))

        sem_lo.append(m - t_ci * s / np.sqrt(c))
        sem_hi.append(m + t_ci * s / np.sqrt(c))

    stats['sem_lo'] = sem_lo
    stats['sem_hi'] = sem_hi

    print(stats)

    yerrList = [stats['mean'] - stats['sem_lo'], stats['sem_hi'] - stats['mean']]
    plt.errorbar(x=xList, y=yList, yerr=yerrList, fmt='none', c='k', elinewidth=2, capsize=5)

    plt.axhline(y=50, color='k', linestyle='--', alpha=0.5)

    plt.ylim((0, 70))
    # plt.xticks(fontsize=14, color='black')
    plt.yticks(fontsize=12, color='black')

    plt.xlabel('', fontsize=16, color='black')
    plt.ylabel('% Reaching the goal-old', fontsize=16, color='black')
    # plt.title('Humans vs RL', fontsize=16, color='black')
    plt.legend(loc='best')
    plt.rcParams['svg.fonttype'] = 'none'
    # plt.savefig('/Users/chengshaozhe/Downloads/exp3a.svg', dpi=600, format='svg')

    plt.show()
