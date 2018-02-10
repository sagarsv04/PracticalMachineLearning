
import pandas as pd
import quandl
import math
import numpy as np
from statistics import mean
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
import os

style.use('ggplot')


def RegressionUsingSKLearn():

    df = quandl.get('WIKI/GOOGL', api_key='6yDydmxh-cSGy9vzbs74')

    impdf = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

    impdf['HL_PCT'] = (impdf['Adj. High'] - impdf['Adj. Close']) / impdf['Adj. Close'] * 100.0

    impdf['PCT_Change'] = (impdf['Adj. Close'] - impdf['Adj. Open']) / impdf['Adj. Open'] * 100.0

    impdf = impdf[['Adj. Close','HL_PCT','PCT_Change','Adj. Volume']]

    forecastCol = 'Adj. Close'

    impdf.fillna(-99999, inplace = True)

    forecastOut = int(math.ceil(0.01*len(impdf)))

    impdf['label'] = impdf[forecastCol].shift(-forecastOut)

    X = np.array(impdf.drop(['label'],1))
    X = preprocessing.scale(X)
    X = X[:-forecastOut]
    type(X)
    len(X)
    XLately = X[-forecastOut:]

    impdf.dropna(inplace=True)
    y = np.array(impdf['label'])
    type(y)
    len(y)
    # X = X[: -forecastOut + 1]
    # find min length and use that much data
    n = min(len(X), len(y))
    # impdf.dropna(inplace=True)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X[:n], y[:n], test_size = 0.2)

    clf = LinearRegression(n_jobs=-1)
    # switch algo
    # clf = svm.SVR()

    clf.fit(X_train, y_train)

    #  Save trained classifier in pickle

    # cwd = os.getcwd()
    # os.chdir(os.getcwd()+'\\Regression')

    with open('LinearRegression.pickle','wb') as f:
        pickle.dump(clf, f)

    #  load trained classifier again
    pickleIn = open('LinearRegression.pickle','rb')
    clf = pickle.load(pickleIn)

    accu = clf.score(X_train,y_train)

    forecastSet = clf.predict(XLately)
    impdf['Forecast'] = np.nan

    lastDate = df.iloc[-1].name
    lastUnix = lastDate.timestamp()
    oneDay = 86400
    nextUnix = lastUnix + oneDay

    for i in forecastSet:
        nextDate = datetime.datetime.fromtimestamp(nextUnix)
        nextUnix += oneDay
        impdf.loc[nextDate] = [np.nan for _ in range(len(impdf.columns)-1)] + [i]

    print(forecastSet, accu, forecastOut)

    # issue solved
    impdf['Adj. Close'].plot()
    impdf['Forecast'].plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')

    return 0;


def bestFitSlope(xs, ys):
    # formula
    m = (((mean(xs) * mean(ys)) - mean(xs * ys)) /
            ((mean(xs)**2) - (mean(xs**2))))
    return m

def bestIntercept(xs, ys, m):
    # formula
    b = mean(ys) - m * mean(xs)
    return b

def predict(m, x, b):
    return (m * x) + b;

def ProgrameBestFitSlope():

    xs = np.array([1,2,3,4,5,6])
    ys = np.array([5,4,6,5,6,7])

    m = bestFitSlope(xs, ys)
    b = bestIntercept(xs, ys, m)

    regline = [(m*x)+b for x in xs]

    style.use('fivethirtyeight')

    predictfor = 8

    predicValue = predict(m, predictfor, b)

    plt.plot(xs,regline)
    plt.scatter(xs,ys, color = 'b')
    plt.scatter(predictfor,predicValue, color = 'r')
    # plt.scatter(xs,regline)
    plt.show()

    return 0;



def main():

    RegressionUsingSKLearn();
    ProgrameBestFitSlope();

    return 0;


if __name__ == '__main__':
    main()
