import pandas as pd
import matplotlib.pyplot as plt
import os
import pylab as pl
import numpy as np
import pickle


def createAllCertainFormatFileList(filePath, fileFormat):
    filenameList = [os.path.join(filePath, relativeFilename) for relativeFilename in os.listdir(filePath)
                    if os.path.isfile(os.path.join(filePath, relativeFilename))
                    if os.path.splitext(relativeFilename)[1] in fileFormat]
    return filenameList


def cleanDataFrame(rawDataFrame):
    cleanConditionDataFrame = rawDataFrame[rawDataFrame.condition != 'None']
    cleanBeanEatenDataFrame = cleanConditionDataFrame[cleanConditionDataFrame.beanEaten != 0]
    return cleanBeanEatenDataFrame

def evalStr(data,columns):
    length=data.shape[1]
    evalResult=[eval(data.iloc[rowIndex,length]) for rowIndex in range(length) ]
    return evalResult

def judgePointEqualToTwoTargets(player,target1,target2):
    if np.linalg.norm(np.array(target1) - np.array(player), ord=1)==\
        np.linalg.norm(np.array(target2) - np.array(player), ord=1):
        return True
    else:
        return False



if __name__ == "__main__":
    resultsPath = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/Results/'
    fileFormat = '.csv'
    resultsFilenameList = createAllCertainFormatFileList(resultsPath, fileFormat)
    policyPath=os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/machinePolicy/'
    policyFile = open(policyPath + "SingleWolfTwoSheepsGrid15.pkl", "rb")
    policy = pickle.load(policyFile)
    resultsDataFrameList = [pd.read_csv(file) for file in resultsFilenameList]
    resultsDataFrame = pd.concat(resultsDataFrameList, sort=False)
    resultsDataFrame = cleanDataFrame(resultsDataFrame)
    participantsTypeList = ['machine' if 'machine' in name else 'Human' for name in resultsDataFrame['name']]
    resultsDataFrame["condition"]=resultsDataFrame["condition"].astype(int)
    wrongIndex=resultsDataFrame[(resultsDataFrame["condition"] ==0) ]
    #wrongIndex.to_csv(resultsPath+"condition0Index"+fileFormat)
    totalTria = wrongIndex.shape[1]
    trajectorys=evalResult=[eval(wrongIndex.iloc[rowIndex,12]) for rowIndex in range(totalTria) ]
    pointNeedToCheckCondition0=[(point,(wrongIndex.iloc[trialIndex,5],\
        wrongIndex.iloc[trialIndex,6]),(wrongIndex.iloc[trialIndex,7],wrongIndex.iloc[trialIndex,8])) for trialIndex in range(totalTria)  for point in trajectorys[trialIndex] \
        if judgePointEqualToTwoTargets(point,(wrongIndex.iloc[trialIndex,5],wrongIndex.iloc[trialIndex,6]),(wrongIndex.iloc[trialIndex,7],wrongIndex.iloc[trialIndex,8]))]
    pointPolicyNeedToCheckCondition0=list()
    for state in pointNeedToCheckCondition0:
        try:
            pointPolicyNeedToCheckCondition0.append(policy[state])
        except KeyError:
            pointPolicyNeedToCheckCondition0.append(policy[(state[0],(state[1][1],state[1][0]))])
    print(pointNeedToCheckCondition0)
    print(pointPolicyNeedToCheckCondition0)





