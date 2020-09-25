from sklearn.datasets import  make_blobs, make_moons,make_circles
from matplotlib import pyplot as plt
from pandas import DataFrame
import numpy as np

def plot_graph(X, y):

    df = DataFrame(dict(x=X[:, 0], x1=X[:, 1], label=y))
    colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'cyan', 4: 'pink'}
    fig, ax = plt.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='x1', label=key, color=colors[key])
    plt.show()
    return X

def genrate_data():
    X, y = make_blobs(n_samples=100, centers=3, n_features=2)
    return X,y


#eps: points having distance less than or equal to eps are considered eps
#min_points: the minimum number of points required to form the density
#seed point: the starting point from where all the distances are to be measured (the centre of the core object)
#labels: a list to of labels assigned to the points
#cluster value: the cluster number


def dbscan(X, seed_point, eps, min_points, labels, cluster_value):
    classes_index = []
    neighbours = []#list to store the neighbours of the seed point having distance less than the epsilon value

    for i in range(0, np.shape(X)[0]):
        if np.linalg.norm(X[seed_point] - X[i]) <= eps:
            classes_index.append(i)
            neighbours.append(X[i])

    #if number of points in neighbourhood is less than min points, then cluster cannot be formed (these are outliers)
    if len(neighbours) < min_points:
         for i in range(len(labels)):
             if i in classes_index:
                 labels[i] = -1

    #else assign the cluster value
    else:
        for i in range(len(labels)):
            if i in classes_index:
                labels[i] = cluster_value

    # print(labels)
    # print(classes_index)
    # print(len(classes_index))

    return labels

X, y = genrate_data()
plot_graph(X,y)
labels = [0]*np.shape(X)[0]
cluster_value = 1

for i in range(0, len(X)):
    if labels[i] == 0:
        labels = dbscan(X, i, 5, 5, labels, cluster_value)
        cluster_value += 1

plot_graph(X,labels)

## references: https://medium.com/@darkprogrammerpb/dbscan-clustering-from-scratch-199c0d8e8da1