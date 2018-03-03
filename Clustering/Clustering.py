
import pandas as pd
import numpy as np
import os, random
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D

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


class K_Means():
    def __init__(self, k=2, tol=0.001, max_iter=300):
        # tol is tollerance ie by how much the centroid gonno move % change
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self,data):

        self.centroids = {}

        for i in range(self.k):
            # i = 0
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    # to print the percentage move with every iteration
                    print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False

            if optimized:
                break

    def predict(self,data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


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

def KMeansFromStrach():

    data = GetData()
    colors = 10*["g","r","c","b","k"]

    clf = K_Means()
    clf.fit(data)

    for centroid in clf.centroids:
        plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
            marker="o", color="k", s=150, linewidths=5)

    for classification in clf.classifications:
        color = colors[classification]
        for featureset in clf.classifications[classification]:
            plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=150, linewidths=5)

    unknowns = np.array([[1,3],
                    [8,9],
                    [0,3],
                    [5,4],
                    [6,4],])

    for unknown in unknowns:
       classification = clf.predict(unknown)
       plt.scatter(unknown[0], unknown[1], marker="*", color=colors[classification], s=150, linewidths=5)


    plt.show()


    return 0


def MeanShiftUsingSKLearn():

    centers = [[1,1,1],[5,5,5],[3,10,10]]

    X, _ = make_blobs(n_samples = 1000, centers = centers, cluster_std = 1.5)

    ms = MeanShift()
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    print(cluster_centers)
    n_clusters_ = len(np.unique(labels))
    print("Number of estimated clusters:", n_clusters_)

    colors = 10*['r','g','b','c','k','y','m']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(X)):
        ax.scatter(X[i][0], X[i][1], X[i][2], c=colors[labels[i]], marker='o')

    ax.scatter(cluster_centers[:,0],cluster_centers[:,1],cluster_centers[:,2],
                marker="x",color='k', s=150, linewidths = 5, zorder=10)

    plt.show()

    return 0


def MeanShiftTitanicDataset():

    df = pd.read_excel('./titanic.xls')
    #print(df.head())
    original_df = pd.DataFrame.copy(df)
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

    clf = MeanShift()
    clf.fit(X)

    labels = clf.labels_
    cluster_centers = clf.cluster_centers_

    original_df['cluster_group'] = np.nan

    for i in range(len(X)):
        # i = 0
        original_df['cluster_group'].iloc[i] = labels[i]

    n_clusters_ = len(np.unique(labels))
    survival_rates = {}
    for i in range(n_clusters_):
        # i = 0
        temp_df = original_df[ (original_df['cluster_group']==float(i)) ]
        #print(temp_df.head())
        survival_cluster = temp_df[  (temp_df['survived'] == 1) ]
        survival_rate = len(survival_cluster) / len(temp_df)
        #print(i,survival_rate)
        survival_rates[i] = survival_rate

    print(survival_rates)

    return 0


def main():
    # KMean is a flat clustering methodology
    # ClusterUsingSKLearn()
    # 70.8938 %
    # Cluste rTitanicDataset()
    # KMeansFromStrach()

    # MeanShift is a hierarchical clustering methodology
    # MeanShiftUsingSKLearn()
    MeanShiftTitanicDataset()

    return 0


if __name__ == '__main__':
    main()
