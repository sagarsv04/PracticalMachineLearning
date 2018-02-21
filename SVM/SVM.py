import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import preprocessing, cross_validation, neighbors, svm
import os, random
from collections import Counter
from math import sqrt
import warnings

# cwd = os.getcwd()
# os.chdir(os.getcwd()+'/SVM')

# Basic SVM Using Breast Cancer Wisconsin data from UCI
def ReadData():
    df = pd.read_csv('../KNN/breastCancerWisconsin.data')
    df.replace('?',-99999,inplace=True)
    # remove coloum id as its useless including it can reduce your accuracy significantly
    df.drop('id',1,inplace=True)
    return df;


def SVMUsingSKLearn():

    df = ReadData()

    # features
    X = np.array(df.drop(['class'],1))
    # labels
    y = np.array(df['class'])

    # cross_validation
    XTrain, XTest, yTrain, yTest = cross_validation.train_test_split(X,y, test_size=0.2)

    clf = svm.SVC()
    clf.fit(XTrain,yTrain)

    accuracy = clf.score(XTest,yTest)
    print("accuracy is %s" %accuracy)

    exampleMeasure = np.array([[4,2,1,1,1,2,3,2,1], [4,2,1,2,2,2,3,2,1]])
    # reshape as well
    exampleMeasure = exampleMeasure.reshape(len(exampleMeasure),-1)

    prediction = clf.predict(exampleMeasure)
    print("prediction for %s" %prediction)

    return 0;

class SVM():
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1:'r',-1:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)

    # train method
    def fit(self, data):
        self.data = data
        # { ||w||: [w,b] }
        optDict = {}
        transforms = [[1,1],
                    [-1,1],
                    [-1,-1],
                    [1,-1]]

        allData = [] # get max and min ranges for graph
        for yi in self.data:
            # yi is class
            for featureset in self.data[yi]:
                for feature in featureset:
                    allData.append(feature)

        self.maxFeatureValue = max(allData)
        self.minFeatureValue = min(allData)
        allData = None # so we are not holding all the data in memory

        stepSizes = [self.maxFeatureValue * 0.1,
                    self.maxFeatureValue * 0.01,
                    # point of expense:
                    self.maxFeatureValue * 0.001,]

        #  extremely expensive
        bRangeMultiple = 5
        #  extremely expensive
        bMultiple = 5
        latestOptimum = self.maxFeatureValue*10

        for step in stepSizes:
            w = np.array([latestOptimum,latestOptimum])
            #  we can do this because convex
            optimized = False
            while not optimized:
                pass

        pass

    def predict(self, features):
        # sign(x.w+b)
        # classification = ???
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)

        return classification

def SVMFromStrach():

    style.use('ggplot')

    # class as key whose values are list of list
    dataDict = {-1:np.array([[1,7],
                            [2,8],
                            [3,8],]),
                1:np.array([[5,1],
                            [6,-1],
                            [7,3],])}


    return 0;



def main():

    SVMUsingSKLearn();
    SVMFromStrach();

    return 0;


if __name__ == '__main__':
    main()
