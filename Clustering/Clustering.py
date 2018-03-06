radius
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
from sklearn.datasets.samples_generator import make_blobs

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


class Mean_Shift(object):
    # radius_norm_step we wanna have a lot of steps in the radius like lot of bandwidths
    def __init__(self, radius = None, radius_norm_step = 100):
        self.radius = radius
        self.radius_norm_step = radius_norm_step

    def fit(self, data):

        if self.radius == None:
            # find centroid of all the data
            all_data_centroid = np.average(data, axis = 0)
            # get magnitude from the origin
            all_data_norm = np.linalg.norm(all_data_centroid)
            # we will use above information to define a radius
            self.radius = all_data_norm / self.radius_norm_step
            print("Radius:",self.radius)

        centroids = {}

        for i in range(len(data)):
            # i = 0
            centroids[i] = data[i]

        weights = [i for i in range(self.radius_norm_step)][::-1]

        while True:
            new_centroids = []
            for i in centroids:
                # i = 0
                in_bandwidth = []
                centroid = centroids[i]

                for featureset in data:
                    # featureset = data[6]

                    # # calculate euclidean norm
                    # if np.linalg.norm(featureset-centroid) < self.radius:
                    #     in_bandwidth.append(featureset)

                    distance = np.linalg.norm(featureset-centroid)
                    if distance == 0:
                        distance = 0.00000000001
                    weight_index = int(distance/self.radius)
                    if weight_index > self.radius_norm_step-1:
                        weight_index = self.radius_norm_step-1

                    to_add = (weights[weight_index]**2)*[featureset]
                    in_bandwidth +=to_add

                new_centroid = np.average(in_bandwidth,axis=0)
                new_centroids.append(tuple(new_centroid))

            uniques = sorted(list(set(new_centroids)))

            to_pop = []

            for i in uniques:
                for ii in [i for i in uniques]:
                    if i == ii:
                        pass
                    elif np.linalg.norm(np.array(i)-np.array(ii)) <= self.radius:
                        #print(np.array(i), np.array(ii))
                        to_pop.append(ii)
                        break

            for i in to_pop:
                try:
                    uniques.remove(i)
                except:
                    pass


            prev_centroids = dict(centroids)

            #  now take only unique centroids
            centroids = {}
            for i in range(len(uniques)):
                # i = 0
                centroids[i] = np.array(uniques[i])

            optimized = True

            for i in centroids:
                # i = 0
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False


            if optimized:
                break

        # save the centroids on train dataset to use with test data
        self.centroids = centroids
        self.classifications = {}

        for i in range(len(self.centroids)):
            self.classifications[i] = []

        for featureset in data:
            #compare distance to either centroid
            distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
            #print(distances)
            classification = (distances.index(min(distances)))

            # featureset that belongs to that cluster
            self.classifications[classification].append(featureset)



    def predict(self,data):
        #compare distance to either centroid
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = (distances.index(min(distances)))
        return classification


def MeanShiftFromStrach():

    data = GetData()
    extra_data = np.array([[8,2],[10,2],[9,3]])
    data = np.concatenate((data, extra_data))

    X, y = make_blobs(n_samples=25, centers=3, n_features=2)
    colors = 10*["g","r","c","b","k"]
    # plt.scatter(data[:, 0],data[:, 1], s=150)

    clf = Mean_Shift()
    # clf.fit(data)
    clf.fit(X)

    centroids = clf.centroids

    # plt.scatter(data[:,0], data[:,1], s=150)

    for classification in clf.classifications:
        color = colors[classification]
        for featureset in clf.classifications[classification]:
            plt.scatter(featureset[0],featureset[1], marker = "x", color=color, s=150, linewidths = 5, zorder = 10)


    for c in centroids:
        plt.scatter(centroids[c][0], centroids[c][1], color='k', marker='*', s=150, linewidths = 5)

    plt.show()


    return 0



def main():
    # KMean is a flat clustering methodology
    # ClusterUsingSKLearn()
    # 70.8938 %
    # Cluste rTitanicDataset()
    # KMeansFromStrach()

    # MeanShift is a hierarchical clustering methodology
    # MeanShiftUsingSKLearn()
    # MeanShiftTitanicDataset()
    MeanShiftFromStrach()

    return 0


if __name__ == '__main__':
    main()
