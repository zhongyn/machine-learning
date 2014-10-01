import numpy as np
import pylab as pl
import perceptron

data = np.loadtxt("twogaussian.csv", delimiter = ',')

###########################################
# classification error of batch perceptron
########################################### 
y = data[:,0]
x = data.copy()
x[:,0] = 1.0
epsilon = 0.001

value = perceptron.batch(x,y,epsilon)
w = value[0]
err = value[1]

#######################################
# linear decision boundary
#######################################
lx = np.linspace(-4,8,50)
ly = -(w[0] + lx*w[1])/w[2]

pl.figure(1)
pl.plot(np.arange(len(err))[1:], err[1:], marker = 'o')
pl.ylim(-2,50)
pl.ylabel("Classification error",fontsize=20)
pl.xlabel("Training epochs",fontsize=20)
pl.savefig("classification_error")

x1 = data[data[:,0]==1]
x2 = data[data[:,0]==-1]

pl.figure(2)
pl.scatter(x1[:,1],x1[:,2],color="blue")
pl.scatter(x2[:,1],x2[:,2],color="red")
pl.plot(lx,ly,linewidth=2)
pl.xlabel("x1",fontsize=20)
pl.ylabel("x2",fontsize=20)
pl.savefig("batch_decision_boundary.png")

pl.show()