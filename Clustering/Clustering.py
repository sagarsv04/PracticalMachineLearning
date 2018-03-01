import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import numpy as np
import os, random
from sklearn import preprocessing
from sklearn.cluster import KMeans
style.use('ggplot')

#https://pythonprogramming.net/static/downloads/machine-learning-data/titanic.xls


'''
This are valuable traits and information on people to determine
whether or not they would survive a sinking of the titanic

Pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
survival Survival (0 = No; 1 = Yes)
name Name
sex Sex
age Age
sibsp Number of Siblings/Spouses Aboard
parch Number of Parents/Children Aboard
ticket Ticket Number
fare Passenger Fare (British pound)
cabin Cabin
embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
boat Lifeboat
body Body Identification Number
home.dest Home/Destination
'''

# cwd = os.getcwd()
# os.chdir(os.getcwd()+'\\Clustering')

def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        # column = columns[2]
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            # 1st column is 'pclass' ie. integer
            # Search for datatype other than int64, float64 in our case a dtype('O')
            # To know more on datatype https://stackoverflow.com/questions/37561991/what-does-a-dtype-of-o-mean
            column_contents = df[column].values.tolist()
            # take different types of content
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                # 'male' in unique_elements
                # unique = 'male'
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df


def ClusterTitanicDataset():

    df = pd.read_excel('./titanic.xls')
    #print(df.head())
    df.drop(['body','name'], 1, inplace=True)
    df.convert_objects(convert_numeric=True)
    df.fillna(0, inplace=True)
    #print(df.head())
    df = handle_non_numerical_data(df)
    # print(df.head())

    X = np.array(df.drop(['survived'], 1).astype(float))
    # se what preprocessing does to data
    # X = preprocessing.scale(GetData())
    X = preprocessing.scale(X)

    y = np.array(df['survived'])

    clf = KMeans(n_clusters=2)
    clf.fit(X)

    correct = 0
    for i in range(len(X)):
        # i = 0
        predict_me = np.array(X[i].astype(float))
        predict_me = predict_me.reshape(-1, len(predict_me))
        prediction = clf.predict(predict_me)
        if prediction[0] == y[i]:
            correct += 1

    print(correct/len(X))

    return 0


def GetData():

    retX = np.array([[1, 2],
                  [1.5, 1.8],
                  [5, 8],
                  [8, 8],
                  [1, 0.6],
                  [9, 11]])

    return retX

# retX = GetData()
# retX = preprocessing.scale(GetData())
# plt.scatter(retX[:, 0],retX[:, 1], s=150, linewidths = 5, zorder = 10)
# plt.show()


def ClusterUsingSKLearn():

    clf = KMeans(n_clusters=2)
    X = GetData()
    clf.fit(X)

    centroids = clf.cluster_centers_
    labels = clf.labels_

    colors = ["g.","r.","c.","y."]*10

    for i in range(len(X)):
        plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)

    plt.scatter(centroids[:, 0],centroids[:, 1], marker = "x", s=150, linewidths = 5, zorder = 10)
    plt.show()

    return 0


def main():

    # ClusterUsingSKLearn()
    # 70.8938 %
    ClusterTitanicDataset()

    return 0


if __name__ == '__main__':
    main()
