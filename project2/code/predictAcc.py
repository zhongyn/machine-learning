import numpy as np
import featureVector as fvec
import matplotlib.pyplot as pl

predictBer = np.load("predictBer.npy")
predictMul = np.load("predictMul.npy")
predictMulAlp = np.load("predictMulAlp.npy")
predictMulFeatRemove = np.load("predictMulFeatRemove.npy")


testLabel = np.loadtxt("../data/test.label", delimiter=' ', dtype=int)

berAcc = fvec.accPredict(testLabel,predictBer)
mulAcc = fvec.accPredict(testLabel,predictMul)

alpRan = predictMulAlp.shape[0]
mulAlpAcc = np.zeros(alpRan)
for index,item in enumerate(predictMulAlp):
	mulAlpAcc[index] = fvec.accPredict(testLabel,item)


print "berAcc = " + str(berAcc) + "; mulAcc = " + str(mulAcc)
print "mulAlpAcc = " + str(mulAlpAcc)

# Confusion matrix
kkmatBer = fvec.confMatrix(testLabel,predictBer)
kkmatMul = fvec.confMatrix(testLabel,predictMul)
np.savetxt("berConfMat.txt", kkmatBer, fmt='%4d')
np.savetxt("mulConfMat.txt", kkmatMul, fmt='%4d')


alpha = [10**i for i in range(-5,1)]

#fig,ax = plt.subplots(11)
#line, = ax.plot(alpha,mulAlpAcc,marker="o",lw=2)
#ax.set_xscale('log')
#plt.plot(alpha,mulAlpAcc,marker="o",lw=2)

fonts = 18
pl.figure(1)
ax1 = pl.subplot(111)
pl.plot(alpha,mulAlpAcc,marker="o",lw=2)
pl.xscale('log')
pl.xlim(0,1)
pl.ylim(0.77,0.82)
pl.xlabel("Alpha",fontsize=fonts)
pl.ylabel("Accuracy",fontsize=fonts)
#pl.legend(loc="lower right")

pl.show()

pl.savefig("alpha.png")