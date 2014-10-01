import numpy as np
import kmeans as km


data = np.loadtxt("../data/data1.csv", delimiter=",")	
numCluster = 5
runs = 200
centers = np.empty([runs,numCluster,2])
sumSqDist = np.empty(runs)

for i in range(runs):
	result = km.kmeans(data, numCluster)
	sumSqDist[i] = result[1]
	centers[i] = result[0]
	print i

np.savez("centersAndSumSqDist.npz", centers=centers, sumSqDist=sumSqDist)
