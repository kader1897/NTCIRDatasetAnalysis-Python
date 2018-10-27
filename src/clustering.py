import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering, AffinityPropagation
from sklearn.preprocessing import normalize

fileName = "data\\clusteringData.txt"


def printResults(header, activityList, predictionList):
    resultTable = {'transport': [0, 0, 0], 'walking': [0, 0, 0], 'airplane': [0, 0, 0]}
    for i in range(len(activityList)):
        resultTable[activityList[i]][predictionList[i]] += 1
    print(header)
    print('****************************')
    print(resultTable)
    print('****************************')
    print()


with open(fileName) as f:
    content = f.readlines()

content = [x.strip().split(",") for x in content]
activityContent = [x for x in content if x[1] != "null"]

for data in [activityContent]:
    imageList = [x[0] for x in data]
    activityList = [None if (x[1] == 'null') else x[1] for x in data]
    labelList = [','.join(x[2:]) for x in data]

    def tokenize(txt):
        return txt.split(",")


    vec = CountVectorizer(tokenizer=tokenize)
    transformed_data = vec.fit_transform(labelList).toarray()
    normalized_data = normalize(transformed_data, norm='l1')
    kmeans = KMeans(n_clusters=3, random_state=0, max_iter=500, n_init=20)
    prediction = kmeans.fit(normalized_data).labels_
    printResults('K-Means Clustering Algorithm', activityList, prediction)

    algorithm = SpectralClustering(n_clusters=3)
    prediction = algorithm.fit(normalized_data).labels_
    printResults('Spectral Clustering Algorithm', activityList, prediction)

    algorithm = AgglomerativeClustering(n_clusters=3)
    prediction = algorithm.fit(normalized_data).labels_
    printResults('Agglomerative Clustering Algorithm', activityList, prediction)

    algorithm = DBSCAN(min_samples=300)
    prediction = algorithm.fit(normalized_data).labels_
    printResults('DBSCAN Clustering Algorithm', activityList, prediction)




