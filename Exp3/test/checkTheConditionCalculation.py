import pandas as pd
import matplotlib.pyplot as plt
import os
import pylab as pl
import numpy as np


def createAllCertainFormatFileList(filePath, fileFormat):
    filenameList = [os.path.join(filePath, relativeFilename) for relativeFilename in os.listdir(filePath)
                    if os.path.isfile(os.path.join(filePath, relativeFilename))
                    if os.path.splitext(relativeFilename)[1] in fileFormat]
    return filenameList


def cleanDataFrame(rawDataFrame):
    cleanConditionDataFrame = rawDataFrame[rawDataFrame.condition != 'None']
    cleanBeanEatenDataFrame = cleanConditionDataFrame[cleanConditionDataFrame.beanEaten != 0]
    return cleanBeanEatenDataFrame


if __name__ == "__main__":
    resultsPath = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/Results/'
    fileFormat = '.csv'
    resultsFilenameList = createAllCertainFormatFileList(resultsPath, fileFormat)
    resultsDataFrameList = [pd.read_csv(file) for file in resultsFilenameList]
    resultsDataFrame = pd.concat(resultsDataFrameList, sort=False)
    resultsDataFrame = cleanDataFrame(resultsDataFrame)
    participantsTypeList = ['machine' if 'machine' in name else 'Human' for name in resultsDataFrame['name']]
    resultsDataFrame["condition"]=resultsDataFrame["condition"].astype(int)
    resultsDataFrame["realCondition"]=(resultsDataFrame['bean2GridX']-resultsDataFrame["playerGridX"]).abs()+\
                                      (resultsDataFrame['bean2GridY']-resultsDataFrame["playerGridY"]).abs()\
                                        -(resultsDataFrame['bean1GridX']-resultsDataFrame["playerGridX"]).abs()-\
                                      (resultsDataFrame['bean1GridY']-resultsDataFrame["playerGridY"]).abs()
    resultsDataFrame["conditionError"]=resultsDataFrame["condition"]==resultsDataFrame["realCondition"]
    wrongIndex = resultsDataFrame[(resultsDataFrame["conditionError"]==False)]
    wrongIndex.to_csv(resultsPath + "allData" + fileFormat)





