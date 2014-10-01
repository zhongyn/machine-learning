import matplotlib.pyplot as pl
import numpy as np


data = np.loadtxt("../data/data1.csv", delimiter=",")	

with np.load("centersAndSumSqDist.npz") as result:
	centers = result["centers"]
	sumSqDist = result["sumSqDist"]

minDist = np.amin(sumSqDist)
maxDist = np.amax(sumSqDist)
meanDist = np.mean(sumSqDist)
stdDist = np.std(sumSqDist)
dist = np.array([minDist,maxDist,meanDist,stdDist])
ticks = np.arange(4)

fonts = 18
pl.figure(1)
ax1 = pl.subplot(111)
pl.scatter(data[:,0],data[:,1],color='b',alpha=0.5,label="data")
pl.scatter(centers[:,:,0],centers[:,:,1],color='r',marker='s',s=22,label='centers')
pl.xlabel("x1",fontsize=fonts)
pl.ylabel("x2",fontsize=fonts)
pl.legend(scatterpoints=1,loc="upper right",fancybox=True)

wid=0.8
fig,ax2 = pl.subplots()
ax2.bar(ticks,dist,width = wid, bottom = 0, color = 'blue')
ax2.set_xticks(ticks+wid/2)
ax2.set_xticklabels(['Mininum','Maximun','Mean','Standard deviation'],size=15)
ax2.set_ylabel("Sum of squared distance",size=16)



pl.show()

# pl.savefig("alpha.png")
