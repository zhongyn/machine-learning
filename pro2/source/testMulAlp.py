import numpy as np
import featureVector as fvec


with np.load("learnMulAlp.npz") as data:
	groups = data["params"][0]
	voca = data["params"][1]
	probIyMulAlpha = data["probIyMulAlpha"]
	probY = data["probY"]

testData = np.loadtxt("../data/test.data", delimiter=' ', dtype=int)
#testLabel = np.loadtxt("../data/test.label", delimiter=' ', dtype=int)

# Generate feature vector for each doc
testMulFeat = fvec.mulFeatGen(testData, voca)

# Number of testing docs
numDocs = len(testMulFeat)


# Apply Navie Bayes classifier to test data
# Multinomial: log(p(x|y)) = sum(xi*log(Pi|y)) + log(p(y))
# p(y=label|x) ~ p(y=label)*p(x|y=label)
alpRan = probIyMulAlpha.shape[0]
probYXmul = np.zeros((alpRan,numDocs,groups))
for i,alp in enumerate(probIyMulAlpha):
	for index, item in enumerate(testMulFeat):
		probYXmul[i,index,:] = np.sum(item*np.log(alp),axis=1)+np.log(probY)


# Find the best prediction label for each doc
predictMulAlp = np.argmax(probYXmul, axis=2)+1

np.save("predictMulAlp.npy", predictMulAlp)