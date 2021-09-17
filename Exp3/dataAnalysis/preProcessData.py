import pandas as pd
import os
import glob
DIRNAME = os.path.dirname(__file__)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
from scipy.stats import ttest_ind
import seaborn as sns
import math
import researchpy

import sys
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..')))
from dataAnalysis import *


def isEatOld(beanEaten):
    if beanEaten == 1:
        return True
    else:
        return False


if __name__ == '__main__':
    resultsPath = os.path.join(os.path.join(DIRNAME, '..'), 'results')

    participants = ['human', 'RL']
    dataPaths = [os.path.join(resultsPath, participant) for participant in participants]
    dfList = [pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False) for dataPath in dataPaths]
    df = pd.concat(dfList, sort=True)
    df = df.rename(columns={'Unnamed: 0': 'trialIndex'})

    totalTrials = max(df["trialIndex"])
    numOfRestTrial = 6
    restTrialInterval = math.ceil(totalTrials / numOfRestTrial)
    restTrials = list(range(0, totalTrials, restTrialInterval))
    df = df[~df.trialIndex.isin(restTrials)]

    # print(df)

    df.condition = df.apply(lambda x: int(x['condition']), axis=1)
    df["eatOld"] = df.apply(lambda x: isEatOld(x['beanEaten']), axis=1)

    df['participantsType'] = ['MEU' if 'max' in name else 'Humans' for name in df['name']]
    df['responseTime'] = df.apply(lambda x: np.mean(eval(x['reactionTime'])[0]), axis=1)

    # df = df[df['condition'] == 0]
    statsTable = researchpy.summary_cont(df["eatOld"])
    # print(statsTable)
    # print(researchpy.crosstab(df['name'], df['condition']))

    numTrialPerConditionToUse = 40

    nameList = df['name'].unique()
    conditionList = df['condition'].unique()

    combineDF = pd.DataFrame()
    for name in nameList:
        for condition in conditionList:
            dfToUse = df[(df.name == name) & (df.condition == condition)]
            dfToUse = dfToUse.iloc[:numTrialPerConditionToUse, :]
            combineDF = pd.concat([combineDF, dfToUse])
        # combineDF = combineDF.sort_values(by=['trialIndex'])
    combineDF = combineDF.sort_values(by=['name', 'trialIndex'])

    # print(researchpy.crosstab(combineDF['name'], combineDF['condition']))

    df = combineDF
    useColNameList = ['trialIndex', 'aimAction', 'bean1GridX', 'bean1GridY',
                      'bean2GridX', 'bean2GridY', 'beanEaten', 'condition', 'goal', 'name',
                      'playerGridX', 'playerGridY', 'trajectory', 'eatOld',
                      'participantsType', 'responseTime']
    df = df.loc[:, useColNameList]
    # df.to_csv(os.path.join(resultsPath, "all/preprocessedData.csv"))

# test first step resonse time
    # df = df[df.condition == 0]
    # eatOld = df[df['eatOld'] == 1]  # old
    # eatNew = df[df['eatOld'] == 0]
    # old = eatOld.groupby('name')['responseTime'].mean()
    # new = eatNew.groupby('name')['responseTime'].mean()
    # des, res = researchpy.ttest(old, new, paired=True)
    # print(des)
    # print(res)

# Fig.2 test and plot commitment
    # df = df[df.condition == 0]

    statDF = pd.DataFrame()
    statDF['eatOldRatio'] = df.groupby(['name', 'condition', 'participantsType'])["eatOld"].mean()
    statDF = statDF.reset_index()

    CItable = researchpy.summary_cont(statDF.groupby(['condition', 'participantsType'])['eatOldRatio'])
    print(CItable)

    sns.set_theme(style="white")
    plt.rcParams['figure.dpi'] = 200
    colorList = [(0.8392156862745098, 0.15294117647058825, 0.1568627450980392),  # red
                 (0.12156862745098039, 0.4666666666666667, 0.7058823529411765)]
    ax = sns.barplot(x="condition", y="eatOldRatio", hue="participantsType", hue_order=['Humans', 'MEU'], data=statDF, errwidth=1, capsize=.1, ci=95, palette=colorList)
    plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.5)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # remove legend title
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[:], labels=labels[:])

    import statsmodels.api as sm
    for i, participantsType in enumerate(['Humans', 'MEU']):
        dfLogit = df[df.participantsType == participantsType]
        Xtrain = dfLogit['condition']
        ytrain = dfLogit[['eatOld']]
        Xtrain = sm.add_constant(Xtrain)

        log_reg = sm.Logit(ytrain, Xtrain).fit()
        print(log_reg.summary())

        Xtest = np.linspace(-5, 5, 500)
        XtestWithInter = sm.add_constant(Xtest)
        predicts = log_reg.predict(XtestWithInter)

        axNew = ax.twiny()
        axNew.plot(Xtest, predicts, color=colorList[i], linestyle='-')
        axNew.set_axis_off()

    # plt.show()
    gg

# Fig.3 test goal infer and plot
    df = df[df.participantsType == 'MEU']
    # df = df[df.participantsType == 'Humans']
    df = df[df.condition == 0]

    # df = df[abs(df.condition) < 3]

    import pickle
    from scipy.interpolate import interp1d

    machinePolicyPath = os.path.abspath(os.path.join(os.path.join(os.getcwd(), '../..'), 'machinePolicy'))
    Q_dict = pickle.load(open(os.path.join(machinePolicyPath, "noise0commitSnakeGoalGird15_policy.pkl"), "rb"))

    softmaxBeta = 2.5
    softmaxPolicy = SoftmaxPolicy(Q_dict, softmaxBeta)
    initPrior = [0.5, 0.5]
    goalInfernce = GoalInfernce(initPrior, softmaxPolicy)

    df['goalPosteriorList'] = df.apply(lambda x: goalInfernce(eval(x['trajectory']), eval(x['aimAction']), (x['bean1GridX'], x['bean1GridY']), (x['bean2GridX'], x['bean2GridY'])), axis=1)
    xnew = np.linspace(0., 1., 20)
    df['goalPosterior'] = df.apply(lambda x: calPosterior(x['goalPosteriorList'], xnew), axis=1)

    statDF = pd.DataFrame()
    statDF['goalPosterior'] = df.groupby(['name', 'beanEaten']).apply(arrMean, 'goalPosterior')
    statDF = statDF.reset_index()
    eatOldDf = statDF[statDF['beanEaten'] == 1]
    eatNewDf = statDF[statDF['beanEaten'] == 2]

    oldPosterior = np.array(eatOldDf['goalPosterior'].tolist())
    newPosterior = np.array(eatNewDf['goalPosterior'].tolist())

    import mne
    from mne import io
    from mne.stats import permutation_cluster_test
    T_obs, clusters, cluster_p_values, H0 = \
        permutation_cluster_test([oldPosterior, newPosterior], n_permutations=1000, tail=1, out_type='mask')
    print(T_obs)
    print(clusters)
    print(cluster_p_values)

    statsList = []
    stdList = []
    statDFList = []
    for goalPosteriorArrMean in [oldPosterior, newPosterior]:
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
    color = (0.56, 0.87, 0.34)  # green
    colorList = [sns.dark_palette(color)[1], sns.dark_palette(color)[-1]]
    fig, ax = plt.subplots()
    plt.rcParams['figure.dpi'] = 200
    xnew = xnew * 100
    lables = ['Goal-old', 'Goal-new']

    from scipy.stats import t, norm
    alpha = 0.05
    t_ci = t.ppf(1 - alpha / 2, degreeOfFreedom)  # t(19)=2.093
    for i in range(len(statsList)):
        ci95 = t_ci * stdList[i]
        plt.plot(xnew, statsList[i], label=lables[i], color=colorList[i])
        plt.fill(np.concatenate([xnew, xnew[::-1]]), np.concatenate([statsList[i] - ci95, (statsList[i] + ci95)[::-1]]), alpha=.3, color=colorList[i])

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.tick_params(top=False, bottom=False, left=True, right=False)

    plt.ylim((0.5, 1.01))
    plt.legend(loc='lower right', fontsize=16)
    plt.legend(loc='best', fontsize=12)
    plt.xlabel('Percentage of trajectory (%)', fontsize=14, color='black')
    plt.ylabel('Posterior probability of the reached goal', fontsize=16, color='black')
    # plt.xticks(np.arange(6), ("0%", "20%", "40%", "60%", "80%", "100%"), fontsize=12, color='black')
    plt.xticks(fontsize=12, color='black')
    plt.yticks(fontsize=12, color='black')
    plt.rcParams['svg.fonttype'] = 'none'
    plt.savefig('/Users/chengshaozhe/Downloads/expS3b0.svg', dpi=600, format='svg')

    plt.show()
