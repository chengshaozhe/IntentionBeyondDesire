import pandas as pd
import os
import glob
DIRNAME = os.path.dirname(__file__)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
from scipy.stats import ttest_ind
from collections import Counter
import researchpy
import sys
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..')))
from dataAnalysis import *

if __name__ == '__main__':
    resultsPath = os.path.join(os.path.join(DIRNAME, '..'), 'results')
    # participants = ['human', 'RL']
    participants = ['all']

    dataPaths = [os.path.join(resultsPath, participant) for participant in participants]
    dfList = [pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False) for dataPath in dataPaths]
    df = pd.concat(dfList, sort=True)
    df['totalSteps'] = df.apply(lambda x: len(eval(x['trajectory'])), axis=1)

    statDF = pd.DataFrame()
    statDF['totalSteps'] = df.groupby(['participantsType', 'name'])["totalSteps"].mean()

    statsTable = researchpy.summary_cont(statDF.groupby(['participantsType'])['totalSteps'])
    print(statsTable)
