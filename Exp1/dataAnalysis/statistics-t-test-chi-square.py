import pandas as pd
import os
import glob
DIRNAME = os.path.dirname(__file__)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
from scipy.stats import ttest_ind
import researchpy
import sys
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..')))

from src.dataAnalysis import *

if __name__ == '__main__':
    resultsPath = os.path.join(os.path.join(DIRNAME, '..'), 'results')
    participants = ['human', 'RL']
    dataPaths = [os.path.join(resultsPath, participant) for participant in participants]
    dfList = [pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False) for dataPath in dataPaths]

    df = pd.concat(dfList, sort=True)
    df['participantsType'] = ['RL' if 'max' in name else 'Humans' for name in df['name']]
    df['trialType'] = ['Critical Disruption' if trial == "special" else 'Random Disruptions' for trial in df['noiseNumber']]

    df["firstIntentionConsistFinalGoal"] = df.apply(lambda x: calculateFirstIntentionConsistency(eval(x['goal'])), axis=1)

    statDF = pd.DataFrame()
    statDF['commitmentRatio'] = df.groupby(['name', 'trialType', 'participantsType'], sort=False)["firstIntentionConsistFinalGoal"].mean()
    statDF['commitmentRatio'] = statDF.apply(lambda x: int(x["commitmentRatio"] * 100), axis=1)

    statDF = statDF.reset_index()

# t-test
    humanDF = statDF[(statDF.participantsType == "Humans") & (statDF.trialType == 'Random Disruptions')]
    rLDF = statDF[(statDF.participantsType == "RL") & (statDF.trialType == 'Random Disruptions')]
    des, res = researchpy.ttest(humanDF['commitmentRatio'], rLDF['commitmentRatio'])
    print(des)
    print(res)

    humanDF = statDF[(statDF.participantsType == "Humans") & (statDF.trialType == 'Critical Disruption')]
    rLDF = statDF[(statDF.participantsType == "RL") & (statDF.trialType == 'Critical Disruption')]
    des2, res2 = researchpy.ttest(humanDF['commitmentRatio'], rLDF['commitmentRatio'])
    print(des2)
    print(res2)

# chi-squre
    dfNormailTrail = df[df['noiseNumber'] != 'special']
    dfSpecialTrail = df[df['noiseNumber'] == 'special']

    resultDf = dfSpecialTrail
    crosstab3, res3 = researchpy.crosstab(resultDf['participantsType'], resultDf['firstIntentionConsistFinalGoal'], test="fisher")
    print(crosstab3)
    print(res3)

# Fisher’s exact test, P < 0.0001, VCramer = 0.73, N = 100
