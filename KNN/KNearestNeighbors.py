import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import preprocessing, cross_validation, neighbors
import os
from collections import Counter
from math import sqrt
import warnings

# cwd = os.getcwd()
# os.chdir(os.getcwd()+'\\KNN')

# Basic KNN Using Breast Cancer Wisconsin data from UCI
def KNNUsingSKLearn():

    df = pd.read_csv('breastCancerWisconsin.data')
    df.replace('?',-99999,inplace=True)
    # remove coloum id as its useless including it can reduce your accuracy significantly
    df.drop('id',1,inplace=True)

    # features
    X = np.array(df.drop(['class'],1))
    # labels
    y = np.array(df['class'])

    # cross_validation
    XTrain, XTest, yTrain, yTest = cross_validation.train_test_split(X,y, test_size=0.2)

    clf = neighbors.KNeighborsClassifier()
    clf.fit(XTrain,yTrain)

    # in KNN ista actually confidence
    accuracy = clf.score(XTest,yTest)
    print("accuracy is %s" %accuracy)

    exampleMeasure = np.array([4,2,1,1,1,2,3,2,1])
    # reshape as well
    exampleMeasure = exampleMeasure.reshape(len(exampleMeasure),-1)

    prediction = clf.predict(exampleMeasure)
    print("prediction for %s" %prediction)

    return 0;

def kNNAlgo(data, predict, k=3):
    # data,predict = dataset,newFeature
    if len(data)>=k:
        warnings.warn('K is set to a value less that total voting groups, Idiot!')

    distance = []
    for group in data:
        # group = 'k'
        for features in data[group]:
            # features = data[group][0]
            euclideanDist = np.linalg.norm(np.array(features)-np.array(predict))
            distance.append([euclideanDist,group])

    # return first k values
    votes = [i[1] for i in sorted(distance)[:k]]
    voteResult = Counter(votes).most_common(1)[0][0]

    return voteResult


def PlotData():
    # plotOne = [1,2]
    # plotTwo = [5,7]
    # euclideanDist = sqrt((plotOne[0] - plotTwo[0])**2 + (plotOne[1] - plotTwo[1])**2)
    # print(euclideanDist)
    style.use('fivethirtyeight')

    # two classes and there features
    dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}

    newFeature = [5,7]
    [[plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]]for i in dataset]

    result = kNNAlgo(dataset,newFeature,k=3)

    plt.scatter(newFeature[0],newFeature[1],s=100,color=result)

    plt.show()


    print(result)

    return 0;


def main():

    KNNUsingSKLearn();
    PlotData();

    return 0;


if __name__ == '__main__':
    main()
