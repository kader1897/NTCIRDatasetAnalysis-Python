import numpy
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

fileName = "data\\groupedImgsWithLabels.txt"

# activities = {
#     "bathroom": 0,
#     "driving": 1,
#     "exercise": 2,
#     "kitchen": 3,
#     "resting": 4,
#     "shopping": 5,
#     "transport": 6,
#     "walking": 7,
#     "work": 8
# }

activities = {}


def printResults(header, predictionList):
    print(header)
    print('****************************')
    activityDict = {}
    for a in predictionList:
        if activityDict.keys().__contains__(a):
            activityDict[a] += 1
        else:
            activityDict[a] = 1

    for a in activityDict:
        print(str(a) + ": " + str(activityDict[a]))

    print('****************************')
    print()


with open(fileName) as f:
    content = f.readlines()

content = [x.strip().split(",") for x in content]

for data in [content]:
    imageList = [x[0] for x in data]
    activityList = [x[-1] for x in data]
    labelList = [','.join(x[1:-2]) for x in data]

activityId = 0
for act in activityList:
    if not activities.__contains__(act):
        activities[act] = activityId
        activityId += 1


def tokenize(txt):
    return txt.split(",")


printResults("Activity List", activityList)
vec = CountVectorizer(tokenizer=tokenize)
transformed_data = vec.fit_transform(labelList).toarray()
normalized_data = normalize(transformed_data, norm='l1')
kmeans = KMeans(n_clusters=9, random_state=0, max_iter=500, n_init=20)
prediction = kmeans.fit(normalized_data).labels_
printResults('K-Means Clustering Algorithm', prediction)

algorithm = SpectralClustering(n_clusters=9)
prediction = algorithm.fit(normalized_data).labels_
printResults('Spectral Clustering Algorithm', prediction)

algorithm = AgglomerativeClustering(n_clusters=9)
prediction = algorithm.fit(normalized_data).labels_
printResults('Agglomerative Clustering Algorithm', prediction)
#
# algorithm = DBSCAN(min_samples=300)
# prediction = algorithm.fit(normalized_data).labels_
# printResults('DBSCAN Clustering Algorithm', activityList, prediction)




