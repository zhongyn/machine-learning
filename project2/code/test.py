import numpy as np
import featureVector as fvec


with np.load("learn.npz") as data:
	groups = data["params"][0]
	voca = data["params"][1]
	probIyBer = data["probIyBer"]
	probIyMul = data["probIyMul"]
	probY = data["probY"]

testData = np.loadtxt("../data/test.data", delimiter=' ', dtype=int)
testLabel = np.loadtxt("../data/test.label", delimiter=' ', dtype=int)

# Generate feature vector for each doc
testFeat = fvec.featureGen(testData, voca)
testBerFeat= testFeat[0]
testMulFeat= testFeat[1]

# Number of testing docs
numDocs = len(testBerFeat)


# Apply Navie Bayes classifier to test data
# Bernoulli: p(x|y) = sum(xi*log(Pi|y)+(1-xi)*log(1-Pi|y))
# p(y=label|x) ~ p(y=label)*p(x|y=label)
# probXYber is 2-d array containing the predictions of p(x|y). A row for one doc, and a column for one label.
# probYXber contains the predictions of p(y|x). 
probXYber = np.zeros((numDocs,groups))
probYXber = np.zeros((numDocs,groups))
for index, item in enumerate(testBerFeat):
	probXYber[index,:] = np.sum(item*np.log(probIyBer) + (1-item)*np.log(1-probIyBer), axis=1)
probYXber = probXYber*probY

# Multinomial: p(x|y) = sum(xi*log(Pi|y))
probXYmul = np.zeros((numDocs,groups))
probYXmul = np.zeros((numDocs,groups))
for index, item in enumerate(testMulFeat):
	probXYmul[index,:] = np.sum(item*np.log(probIyMul),axis=1)
probYXmul = probXYmul*probY

# Find the best prediction label for each doc
preditBer = np.argmax(probYXber, axis=1)
predictMul = np.argmax(probYXmul, axis=1)
