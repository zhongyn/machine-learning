import matplotlib.pyplot as pl
import numpy as np

data = np.loadtxt("../data/data2.csv", delimiter=",")	

with np.load("findK.npz") as result:
	k = result["k"]
	minDist = result["minDist"]
	minCenter = result['minCenter']


fig,ax1 = pl.subplots()
fonts = 18
ax1.plot(k,minDist,color='b',marker='o')
ax1.set_xlabel('k',size=16)
ax1.set_ylabel('Minimum Sum of Squared Distance',size=16)

knees = np.array([[2,3],[4,10]])
kindex = knees-2

f,axarr = pl.subplots(2,2)
for i in range(2):
	for j in range(2):
		axarr[i,j].set_title('k = '+str(knees[i,j]))
		axarr[i,j].scatter(data[:,0],data[:,1],color='b',alpha=0.5,label='data')
		axarr[i,j].scatter(minCenter[kindex[i,j]][:,0],minCenter[kindex[i,j]][:,1],color='r',marker='s',s=20,label='centers')
		axarr[i,j].legend(scatterpoints=1,fancybox=True,fontsize=10)

pl.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
pl.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)

pl.show()