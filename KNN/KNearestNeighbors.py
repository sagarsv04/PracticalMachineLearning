import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import preprocessing, cross_validation, neighbors
import os


# cwd = os.getcwd()
# os.chdir(os.getcwd()+'\\KNN')

# Basic KNN Using Breast Cancer Wisconsin data from UCI


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
