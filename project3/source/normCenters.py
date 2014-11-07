import numpy as np
import norm as nm

data = np.loadtxt("../data/data3.csv", delimiter=",")	
centers = nm.centers(data)
np.save("normCenters.npy", centers)


data2 = nm.normdata(data)
centers = nm.centers(data2)
np.save("normCentersNew.npy", centers)

