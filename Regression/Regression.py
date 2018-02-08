
import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

df = quandl.get('WIKI/GOOGL')

impdf = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

impdf['HL_PCT'] = (impdf['Adj. High'] - impdf['Adj. Close']) / impdf['Adj. Close'] * 100.0

impdf['PCT_Change'] = (impdf['Adj. Close'] - impdf['Adj. Open']) / impdf['Adj. Open'] * 100.0

impdf = impdf[['Adj. Close','HL_PCT','PCT_Change','Adj. Volume']]

forecastCol = 'Adj. Close'

impdf.fillna(-99999, inplace = True)

forecastOut = int(math.ceil(0.01*len(impdf)))

impdf['label'] = impdf[forecastCol].shift(-forecastOut)

impdf.dropna(inplace=True)

X = np.array(impdf.drop(['label'],1))
y = np.array(impdf['label'])

X = preprocessing.scale(X)

# X = X[: -forecastOut + 1]

# impdf.dropna(inplace=True)
y = np.array(impdf['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)

clf = LinearRegression()
# switch algo
# clf = svm.SVR()

clf.fit(X_train, y_train)

accu = clf.score(X_train,y_train)
