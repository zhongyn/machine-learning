import numpy as np
import kmeans as km
import matplotlib.pyplot as pl

def centers(data):

	numCluster = 2
	runs = 200
	centers = np.empty([runs,numCluster,2])
	sumSqDist = np.empty(runs)

	for i in range(runs):
		result = km.kmeans(data, numCluster)
		sumSqDist[i] = result[1]
		centers[i] = result[0]
		print i
		
	return centers


def normplot(data,centers):

	fonts = 15
	fig,ax = pl.subplots()
	ax.scatter(data[:,0],data[:,1],color='b',alpha=0.5,label="data")
	ax.scatter(centers[:,:,0],centers[:,:,1],color='r',marker='s',s=22,label='centers')
	ax.set_xlabel("x1",fontsize=fonts)
	ax.set_ylabel("x2",fontsize=fonts)
	ax.legend(scatterpoints=1,loc="upper right",fancybox=True)
	pl.show()


def normdata(data):

	mean = np.mean(data,axis=0)
	data1 = data-mean
	var = np.var(data1,axis=0)
	data2 = data1/np.sqrt(var)
	return data2