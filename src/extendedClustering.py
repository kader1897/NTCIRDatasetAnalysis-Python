import numpy as np
import os

from sklearn.cluster import KMeans

print("Loading image features data...")
data = np.load("..\\arrays\\u2ImageFeaturesResnet50.npy")


# Load images and matching activity as two arrays
fileName = "..\\data\\groupedImgsWithLabels.txt"
with open(fileName) as f:
    content = f.readlines()

content = [x.strip().split(",") for x in content]
for data in [content]:
    imageList = [x[0] for x in data]
    activityList = [x[-1] for x in data]

# Load file names corresponding to image features
static_text = "D:\\kader\\kad_ER_\\METU\\CENG (Grad)\\Master's Thesis\\NTCIR13-lifelog2-phase2-images\\Images\\"
with open("\\..\\data\\u2AllImages.txt") as f:
    str_content = f.readlines()

str_content = [x.strip().replace(static_text,"") for x in str_content]

n_clusters = 9
kmeans = KMeans(n_clusters=n_clusters, verbose=1).fit(data)

for i in range(n_clusters):
    tempList = np.where(kmeans == i)


np.savetxt("clustering.txt", kmeans.labels_)

