import numpy as np
import featureVector as fvec


with np.load("learnBer.npz") as data:
	groups = data["params"][0]
	voca = data["params"][1]
	probIyBer = data["probIyBer"]
	probY = data["probY"]

testData = np.loadtxt("../data/test.data", delimiter=' ', dtype=int)

# Generate feature vector for each doc
testBerFeat = fvec.berFeatGen(testData, voca)
print "Feature Generated"

# Number of testing docs
numDocs = len(testBerFeat)

# Apply Navie Bayes classifier to test data
# Bernoulli: log(p(x|y)) = sum(xi*log(Pi|y)+(1-xi)*log(1-Pi|y))+log(P(y))
# p(y=label|x) ~ p(y=label)*p(x|y=label)
# probXYber is 2-d array containing the predictions of p(y|x). A row for one doc, and a column for one label.

probYXber = np.zeros((numDocs,groups))
for index, item in enumerate(testBerFeat):
	probYXber[index,:] = np.sum(item*np.log(probIyBer) + (1-item)*np.log(1-probIyBer), axis=1) + np.log(probY)


# Find the best prediction label for each doc
# Plus 1 due to index starting from 0
predictBer = np.argmax(probYXber, axis=1)+1

np.save("predictBer.npy", predictBer)