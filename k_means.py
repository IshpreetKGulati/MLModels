from sklearn.datasets import  make_blobs, make_moons,make_circles
from matplotlib import pyplot as plt
from pandas import DataFrame
import numpy as np


def generate_data():
    X, y = make_blobs(n_samples=1000, centers=3, n_features=2)
    #X, y = make_moons(n_samples=1000, noise=0.1)
    #X, y = make_circles(n_samples=1000, noise=0.05)

    #scaling (max-min)
    X = (X - np.amin(X, axis=0))/ (np.amax(X, axis=0) - np.amin(X, axis=0))

    df = DataFrame(dict(x=X[:, 0], x1=X[:, 1], label=y))
    colors = {0:'red', 1:'blue', 2:'green', 3:'cyan', 4:'pink'}
    fig, ax = plt.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
       group.plot(ax=ax, kind='scatter', x='x', y='x1', label=key, color=colors[key])
    plt.show()

    return X, y

def k_means(X, epochs, threshold, number_of_clusters ):
    number_of_features = len(X[0])
    centroids = {}

    #centroid initialization : points from data set

    for i in range(number_of_clusters):
        centroids[i] = X[i]

    b = False

    for i in range(epochs):
        classes = {}
        for k in range(0, number_of_clusters):
            classes[k] = []
        for features in X:
            distance = [np.linalg.norm(features - centroids[centroid]) for centroid in centroids]
            classes[distance.index(min(distance))].append(features)
        prev_centroids = dict(centroids)

    #update the values of centroids

        for c in classes:
            centroids[c] = np.average(classes[c], axis=0)

        for centroid in centroids:
            if (prev_centroids[centroid] - centroids[centroid]).all() < threshold:
                b = True

        if b  == True:
            break

    return centroids,classes

#to find the number of clusters using elbow curve
def elbow_curve(centroids, classes):
    distortion = 0
    for c in classes:
        for centroid in classes[c]:
           distortion += np.linalg.norm(centroid - centroids[c])
    return distortion


def dist(x, y):
    sq = np.square(x - y)
    su = np.sum(sq, axis=1)
    d = np.sqrt(su)
    return np.average(d)

# to find the number of clusters using sillhoutte score
def silhoutte_score(classes):
    count = score = 0
    for c in classes:
        x = classes[c]
        for value in classes[c]:
            a = dist(x, value)

        min_inter_cluster_dist = float('inf')
        for c1 in classes:
            if c1 != c:
                inter_cluster_dist = dist(classes[c1], value)
                if  min_inter_cluster_dist > inter_cluster_dist:
                    min_inter_cluster_dist = inter_cluster_dist

        b = min_inter_cluster_dist
        count += 1
        score += (b-a) / max(b,a)

    return score/count #silhoutte score

X, Y = generate_data()
d = []
k = []

#graph for silhoutte score
for i in range(2,11):
    k.append(i)
    centroids, classes = k_means(X, 50, 0.00000001,i)
    score = silhoutte_score(classes)
    print(score)
    d.append(score)

plt.scatter(k,d)
plt.xlabel('number of clusters')
plt.ylabel('silhouette score')
plt.xlabel('number of cluster')
plt.ylabel('silhouette score')
plt.plot(k,d)
plt.show(d)

#graph for elbow curve
for i in range(1,30):
    k.append(i)
    centroids, classes = k_means(X, 50, 0.00000001,i)
    dis = elbow_curve( centroids, classes)
    d.append(dis)

plt.xlabel('number of clusters')
plt.ylabel('distortion')
plt.plot(k,d)
plt.scatter(k,d)
plt.show()

colors = 10 * ["r", "g", "c", "b", "k"]
for c in classes:
    color = colors[c]
    for f in classes[c]:
        plt.scatter(f[0], f[1], color=color, s=30)

for centroid in centroids:
    plt.scatter(centroids[centroid][0], centroids[centroid][1], s=130, marker="x")
plt.show()

