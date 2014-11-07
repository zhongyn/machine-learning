import numpy as np
import kmeans as km

data = np.loadtxt("../data/data2.csv", delimiter=",")	

k = np.arange(2,16)
minDist = np.empty(len(k))
minCenter = []

runs = 10
sumSqDist = np.empty(runs)
for i, item in enumerate(k):
	print i+2
	centers = np.empty([runs,item,2])
	for j in range(runs):
		result = km.kmeans(data,item)
		sumSqDist[j] = result[1]
		centers[j] = result[0]
	minCenter.append(centers[np.argmin(sumSqDist)])
	minDist[i] = np.amin(sumSqDist)
	print minCenter[i]


np.savez("findK.npz", k=k, minDist=minDist, minCenter=minCenter)
