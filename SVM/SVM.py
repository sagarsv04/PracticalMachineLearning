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
        #  we dont need to take as small steps with b as we do with w
        bMultiple = 5
        latestOptimum = self.maxFeatureValue*10

        for step in stepSizes:
            w = np.array([latestOptimum,latestOptimum])
            #  we can do this because convex
            optimized = False
            while not optimized:
                # we can thread this as we are saving it in a dict
                # we are not giving b the same optimizing treatment of big/small step as w
                for b in np.arange(-1*(self.maxFeatureValue*bRangeMultiple),
                                    self.maxFeatureValue*bRangeMultiple,
                                    step*bMultiple):
                    for transformation in transforms:
                        wT = w*transformation
                        foundOption = True
                        # weakest link in SVM fundamentally
                        # SMO attempts to fix this a bit
                        # yi(xi.w+b) >= 1
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi*(np.dot(wT,xi)+b)>=1:
                                    foundOption = False
                                    # break
                        if foundOption:
                            # magnitude of a vector
                            optDict[np.linalg.norm(wT)] = [wT,b]

                if w[0] < 0:
                    optimized= True
                    print('Optimized a step.')
                else:
                    w = w - step

            norms = sorted([n for n in optDict])
            # ||w|| : [w,b]
            optChoice = optDict[norms[0]]
            self.w = optChoice[0]
            self.b = optChoice[1]
            latestOptimum = optChoice[0][0]+step*2

        return 0;

    def predict(self, features):
        # sign(x.w+b)
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])

        return classification

    def visualize(self):
        [[self.ax.scatter(x[0],x[1],s=100,color=self.colors[i]) for x in self.data[i]] for i in self.data]
        # hyperplane = x.w+b
        # v = x.w+b
        # psv = 1
        # nsv = -1
        # dec = 0
        def hyperplane(x,w,b,v):
            return (-w[0]*x-b+v)/w[1]

        datarange = (self.minFeatureValue*0.9,self.maxFeatureValue*1.1)
        hypXmin = datarange[0]
        hypXmax = datarange[1]

        # (w.x+b) = 1
        # positive support vector hyperplane
        # value of y? given x
        psv1 = hyperplane(hypXmin, self.w, self.b, 1)
        psv2 = hyperplane(hypXmax, self.w, self.b, 1)
        # plot x, y
        self.ax.plot([hypXmin,hypXmax],[psv1,psv2])

        # (w.x+b) = -1
        # negative support vector hyperplane
        # value of y? given x
        nsv1 = hyperplane(hypXmin, self.w, self.b, -1)
        nsv2 = hyperplane(hypXmax, self.w, self.b, -1)
        # plot x, y
        self.ax.plot([hypXmin,hypXmax],[nsv1,nsv2])

        # (w.x+b) = 0
        # neutral support vector hyperplane
        # value of y? given x
        zsv1 = hyperplane(hypXmin, self.w, self.b, 0)
        zsv2 = hyperplane(hypXmax, self.w, self.b, 0)
        # plot x, y
        self.ax.plot([hypXmin,hypXmax],[zsv1,zsv2])

        plt.show()

def SVMFromStrach():

    style.use('ggplot')

    # class as key whose values are list of list
    dataDict = {-1:np.array([[1,7],
                            [2,8],
                            [3,8]]),
                1:np.array([[5,1],
                            [6,-1],
                            [7,3]])}

    svm = SVM()
    svm.fit(data=dataDict)

    predictUs = [[0,10],
                [1,3],
                [3,4],
                [3,5],
                [5,5],
                [5,6],
                [6,-5],
                [5,8]]

    for p in predictUs:
        svm.predict(p)

    svm.visualize()



    return 0;



def main():

    # SVMUsingSKLearn();
    SVMFromStrach();

    return 0;


if __name__ == '__main__':
    main()
