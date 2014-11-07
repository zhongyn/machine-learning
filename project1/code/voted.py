import numpy as np
import pylab as pl
import perceptron


data = np.loadtxt("iris-twoclass.csv", delimiter = ',')

################################################
# classification error of original training set 
################################################
y = data[:,0]
x = data.copy()
x[:,0] = 1.0
times = 100
value = perceptron.voted(x,y,times)
w = value[0]
c = value[1]
err = value[2]


#################################################
# classification error of shuffled sets
################################################# 
datasf = data.copy()
sftimes = 30
sferr = np.empty([sftimes,times])

for i in range(sftimes):
	np.random.shuffle(datasf)	
	ys = datasf[:,0]	
	xs = datasf.copy()	
	xs[:,0] = 1.0	
	valuesf = perceptron.voted(xs,ys,times)	
	ws = valuesf[0]	
	cs = valuesf[1]	
	sferr[i] = valuesf[2]	
errs = np.mean(sferr,axis=0)


#################################################
# prediction of sample points
################################################# 
sampnum = 600
xv = np.random.random_sample([sampnum,2])
xv0 = xv[:,0] * 8
xv1 = xv[:,1] * 2.5
xvisua = np.empty((sampnum,3))
xvisua[:,0] = 1.0
xvisua[:,1] = xv0
xvisua[:,2] = xv1
predsamp = perceptron.votepredict(xvisua,w,c)
mask1 = (predsamp == -1)
mask2 = (predsamp == 1)


#################################################
# average perceptron 
################################################# 

wavg = c.dot(w)
xavg = np.linspace(0,8,50)
yavg = -(wavg[0] + xavg*wavg[1])/wavg[2]



###########################################################
# classification errors of original set and shuffled sets.
###########################################################
errx = np.linspace(1,100,100)

pl.figure(1,figsize = (8,8))
fonts = 15
linwid = 2

ax1 = pl.subplot(212)
pl.plot(errx,err,marker = 'o',linewidth=linwid,label="original training set")
pl.ylim(0,70)
pl.xlabel("Training epochs", fontsize=fonts)
pl.ylabel("Classification error",fontsize=fonts)
pl.legend()

ax2 = pl.subplot(211,sharex=ax1)
pl.plot(errx,errs,marker = 'o',linewidth=linwid,label="average of 30-shuffle sets")
pl.ylabel("Classification error",fontsize=fonts)
pl.legend()

pl.savefig("voted_classification_error.png")

################################################################
# scatter plot of training set and average perceptron boundary.
################################################################
pl.figure(2)
pl.scatter(x[:,1][y==1],x[:,2][y==1],color='r')
pl.scatter(x[:,1][y==-1],x[:,2][y==-1],color='b')
pl.plot(xavg,yavg,linewidth=2.5)
pl.xlabel("x1",fontsize=20)
pl.ylabel("x2",fontsize=20)
pl.savefig("voted_aver_decision_boundary.png")


#############################################################
# visualization of decision boundary with sample points.
#############################################################
pl.figure(3)
pl.scatter(xv0[mask1],xv1[mask1],c='b')
pl.scatter(xv0[mask2],xv1[mask2],c='r')
pl.xlabel("x1",fontsize=20)
pl.ylabel("x2",fontsize=20)
pl.ylim(-0.5,3)
pl.savefig("voted_visua_decision_boundary.png")

pl.plot(xavg,yavg,linewidth=2.5)
pl.savefig("voted_decision_boundary.png")


pl.show()

