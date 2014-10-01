import numpy as np
import random

def kmeans(data, numCluster):

	numData = len(data)
	index = random.sample(xrange(numData), numCluster)
	centers = data[index]

	label = np.empty(numData)
	newCenters = np.empty([numCluster,2])

	while True:
		for i, item in enumerate(data):
			dists = np.sqrt(np.sum((item-centers)**2,axis=1))
			label[i] = np.argmin(dists)

		for i in range(numCluster):
			cluster = data[label==i]
			newCenters[i] = np.sum(cluster,axis=0)/len(cluster)

		if np.array_equal(centers,newCenters):
			sumSqDist = 0.0;
			for i in range(numCluster):
				cluster = data[label==i]
				tmp = np.sum(np.sum((centers[i]-cluster)**2,axis=1))
				sumSqDist += tmp
			break
		else:
			centers = newCenters.copy()

	return [centers, sumSqDist]















