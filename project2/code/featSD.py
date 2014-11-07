import numpy as np
import featureVector as fvec


with np.load("learnMulAlp.npz") as data:
	groups = data["params"][0]
	voca = data["params"][1]
	probIyMulAlpha = data["probIyMulAlpha"][-2]
	probY = data["probY"]

featSD = np.std(probIyMulAlpha,axis=0)
mask = (featSD > 1e-6)							

newProbIy = probIyMulAlpha[:,mask]

testData = np.loadtxt("../data/test.data", delimiter=' ', dtype=int)

# Generate feature vector for each doc
testMulFeat = fvec.mulFeatGen(testData, voca)
newFeat = testMulFeat[:,mask]
print "newFeat copied"
del testMulFeat
print "oldFeat deleted"

# Number of testing docs
numDocs = len(newFeat)


# Apply Navie Bayes classifier to test data
# Multinomial: log(p(x|y)) = sum(xi*log(Pi|y)) + log(p(y))
# p(y=label|x) ~ p(y=label)*p(x|y=label)
probYXmul = np.zeros((numDocs,groups))
for index, item in enumerate(newFeat):																																																																																	
	probYXmul[index,:] = np.sum(item*np.log(newProbIy),axis=1)+np.log(probY)


# Find the best prediction label for each doc
predictMulNew = np.argmax(probYXmul, axis=1)+1

np.save("predictMulFeatRemove.npy", predictMulNew)