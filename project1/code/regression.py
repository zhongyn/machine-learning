
import numpy as np
import linreg
import pylab as pl


data = np.loadtxt("train.csv", delimiter = ',')
testData = np.loadtxt("test.csv",delimiter = ',')

x = data[:,:-1]
y = data[:,-1]
testx = testData[:,:-1]
testy = testData[:,-1]

##########################
# Experiment
##########################
lam = [0] + [10**i for i in range(-3,3)]
w = linreg.linreg(x,y,lam) # return linear regression parameters w
sse = linreg.sserr(x,y,w,lam) # return sum-squared error for tranining set
testsse = linreg.sserr(testx,testy,w,lam) # return sum-squared error for testing set
predicterr = linreg.test(testx,testy,w,lam)

#####################################
# remove singular squared errors
#####################################
sings= 8
singus = np.sort(predicterr,axis=1)[:,-sings:]
sinerr = 0
for i in range(sings):
	sinerr += 0.5*singus[:,i]**2
rmsingular = testsse-sinerr


##########################
# Exploration
##########################
filen = 5
sseFilter = np.empty((filen,len(lam)))
filt = [10**i for i in range(-7,-2)]
for index,item in enumerate(filt):
	tem = w.copy()
	tem[abs(tem)<=item] = 0.0
	sseFilter[index] = linreg.sserr(x,y,tem,lam)

fonts = 15
linwid = 2

pl.figure(1,figsize = (9,10))

ax1 = pl.subplot(212)
pl.plot(lam,sse,color="blue",linewidth=linwid,marker='o',label="tranining")
pl.xscale('log')
pl.xlim(0,100)
pl.ylim([100,500])
pl.xlabel("Lambda",fontsize=fonts)
pl.ylabel("Regularized SSE",fontsize=fonts)
pl.legend(loc="lower right")

ax2 = pl.subplot(211,sharex=ax1)
pl.plot(lam,testsse,color="red",linewidth=linwid,marker='o',label="testing")
pl.ylim([100000,1300000])
pl.ylabel("Regularized SSE",fontsize=fonts)
pl.legend()
pl.savefig("linear_regression.png",dpi=120)


pl.figure(2,figsize = (4,5))

bar = pl.bar(np.arange(filen), sseFilter[:,-2], width = 0.8, bottom = 0, color = 'blue', log = True)
pl.ylabel("Regularized SSE",fontsize=fonts)
#bar.tick_params(labelsize = fonts)
pl.xticks([],[])
pl.savefig("filtering_features.png")

pl.figure(3)
pl.plot(lam,rmsingular,color="m",linewidth=linwid,marker='o',label="remove 8 largest squared errors")
#pl.ylim([100000,1300000])
pl.ylabel("Regularized SSE",fontsize=fonts)
pl.legend()
pl.xscale('log')
pl.xlabel("Lambda",fontsize=fonts)

pl.savefig("linreg_singularerr.png",dpi=120)

pl.show()
