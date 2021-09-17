import pandas as pd
import os
import glob
DIRNAME = os.path.dirname(__file__)
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind
import researchpy
import seaborn as sns
import sys
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..')))
from scipy.stats import t, norm

from src.dataAnalysis import *

pd.set_option('max_columns', 8)
pd.set_option('display.width', None)

if __name__ == '__main__':
    resultsPath = os.path.join(os.path.join(DIRNAME, '..'), 'results')

    participants = ['human', 'RL']
    dataPaths = [os.path.join(resultsPath, participant) for participant in participants]
    dfList = [pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False) for dataPath in dataPaths]
    df = pd.concat(dfList, sort=True)

    df['trialType'] = ['Critical Disruption' if trial == "special" else 'Random Disruptions' for trial in df['noiseNumber']]
    df['participantsType'] = ['RL' if 'max' in name else 'Humans' for name in df['name']]

    df["firstIntentionConsistFinalGoal"] = df.apply(lambda x: calculateFirstIntentionConsistency(eval(x['goal'])), axis=1)

    statDF = pd.DataFrame()
    statDF['commitmentRatio'] = df.groupby(['name', 'trialType', 'participantsType'], sort=False)["firstIntentionConsistFinalGoal"].mean()
    statDF['commitmentRatio'] = statDF.apply(lambda x: int(x["commitmentRatio"] * 100), axis=1)

# for demo: non zero point for RL
    # statDF['commitmentRatio'] = statDF.apply(lambda x: max(1, x["commitmentRatio"]), axis=1)

    statDF = statDF.reset_index()
    # print(statDF)
    # statDF.to_csv('statDF.csv')

    CItable = researchpy.summary_cont(statDF.groupby(['trialType', 'participantsType'])['commitmentRatio'])
    print(CItable)


# plotting
    # for demo non zero point for RL
    statDF['commitmentRatio'] = statDF.apply(lambda x: max(1, x["commitmentRatio"]), axis=1)

    sns.set_theme(style="white")
    colorList = [(0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
                 (0.12156862745098039, 0.4666666666666667, 0.7058823529411765)]
    ax = sns.barplot(x='trialType', y="commitmentRatio", hue="participantsType", data=statDF, ci=None, palette=colorList)
    # ax = sns.barplot(x='trialType', y="commitmentRatio", hue="participantsType", data=statDF, ci=None, palette=colorList, edgecolor=(0, 0, 0), linewidth=0)
    # plt.rcParams['figure.figsize'] = (12.0, 4.0)
    plt.rcParams['figure.dpi'] = 300

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

    stats = statDF.groupby(['trialType', 'participantsType'], sort=False)['commitmentRatio'].agg(['mean', 'count', 'std'])
    # print(stats)
    # print('-' * 50)

    ci_hi = []
    ci_lo = []
    for i in stats.index:
        m, c, s = stats.loc[i]
        alpha = 0.05
        t_ci = t.ppf(1 - alpha / 2, 49)  #   two-tailed 95%, N=50
        # print(t_ci)

        ci_lo.append(m - t_ci * s / np.sqrt(c))
        ci_hi.append(m + t_ci * s / np.sqrt(c))

    stats['ci_lo'] = ci_lo
    stats['ci_hi'] = ci_hi
    print(stats)

    yerrList = [stats['mean'] - stats['ci_lo'], stats['ci_hi'] - stats['mean']]
    plt.errorbar(x=xList, y=yList, yerr=yerrList, fmt='none', c='k', elinewidth=2, capsize=5)

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    plt.ylim((0, 101))
    plt.xticks(fontsize=16, color='black')
    plt.yticks(fontsize=14, color='black')

    plt.xlabel('', fontsize=16, color='black')
    plt.ylabel('% Choosing the original destination', fontsize=18, color='black')
    # plt.title('Commitment Ratio', fontsize=fontSize, color='black')
    plt.legend(loc='best', fontsize=14)
    plt.rcParams['svg.fonttype'] = 'none'

    # plt.savefig('/Users/chengshaozhe/Downloads/exp1.svg', dpi=600, format='svg')
    # plt.show()
