import src.utility as util
import numpy as np
import os

from sklearn.cluster import KMeans
from collections import Counter

print("Loading image features data...")
data = np.load("..\\arrays\\u2ImageFeaturesResnet50.npy")
imgActivityDict = {}
imgActivityCounter = {}
imgLabelsDict = {}

# Load images and matching activity as a dictionary
fileName = "..\\data\\groupedImages.txt"
with open(fileName) as f:
    content = f.readlines()

content = [x.strip().split(",") for x in content]
for d in content:
    act = d[-1]
    imgActivityDict[d[0]] = act
    if(imgActivityCounter.__contains__(act)):
        imgActivityCounter[act] += 1
    else:
        imgActivityCounter[act] = 1

# Load file names corresponding to image features
static_text = "D:\\kader\\kad_ER_\\METU\\CENG (Grad)\\Master's Thesis\\NTCIR13-lifelog2-phase2-images\\Images\\"
with open("..\\data\\u2AllImages.txt") as f:
    str_content = f.readlines()

# Activities matching file names. If no activity, mark as None.
str_content = [x.strip().replace(static_text,"").replace("\\","/") for x in str_content]
u2ImgActivityList = [imgActivityDict[x] if imgActivityDict.__contains__(x) else None for x in str_content]

# K-means clustering with 11 clusters
n_clusters = 11
kmeans = KMeans(n_clusters=n_clusters, verbose=1, n_jobs=-1).fit(data)

# Assign each cluster an activity. Activity of the cluster is the most frequent activity inside.
u2ClusteredActivityList = util.clusterAssignmentIgnoreFreq(n_clusters=n_clusters, kmeans_result = kmeans, activityList = u2ImgActivityList)

with open("..\\data\\imgLabels.txt") as f:
    label_content = f.readlines()

# Generate label list for clustering...
label_data = []
labelActivityList = []
for line in label_content:
    seq = line.strip().split(",")
    lst = seq[0].strip().split('/')
    lst.insert(1, lst[1].split('-')[0])
    key = '/'.join(lst)
    imgLabelsDict[key] = seq[1:]


dataMatchDict = {}
index = 0
for x in str_content:
    if imgLabelsDict.__contains__(x):
        label_data.append(','.join(imgLabelsDict.get(x)))
        labelActivityList.append(imgActivityDict[x] if imgActivityDict.__contains__(x) else None)
        dataMatchDict[x] = index
        index += 1


labelData = util.wordToVec(label_data)
kmeansForLabels = KMeans(n_clusters=n_clusters, verbose=1, n_jobs=-1).fit(labelData)

# Assign each cluster an activity. Activity of the cluster is the most frequent activity inside.
u2ClusteredActivityListWrtLabels = util.clusterAssignmentIgnoreFreq(n_clusters=n_clusters, kmeans_result = kmeansForLabels, activityList = labelActivityList)

matchCounter = 0
counter = 0
# Write the result to a file...
f = open("result.txt", "w")
for n in range(len(str_content)):
    imgName = str_content[n]
    activityWrtLabels = u2ClusteredActivityListWrtLabels[dataMatchDict.get(imgName)] if dataMatchDict.__contains__(imgName) else 'None'
    f.write(imgName + ": " + str(u2ImgActivityList[n]) + " - " + u2ClusteredActivityList[n] + " - " + activityWrtLabels + "\n")
    counter += 1
    if u2ClusteredActivityList[n] == activityWrtLabels:
        matchCounter += 1

f.close()
print("Match Counter: " + str(matchCounter) + "\n")
print("Total Counter: " + str(counter) + "\n")
print("Ratio: " + str(matchCounter/(counter * 1.0)) + "\n")


