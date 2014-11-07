import numpy as np
import norm as nm


data = np.loadtxt("../data/data3.csv", delimiter=",")	
dataNew = nm.normdata(data)

centers = np.load("normCenters.npy")
centersNew = np.load("normCentersNew.npy")

nm.normplot(data,centers)
nm.normplot(dataNew, centersNew)
