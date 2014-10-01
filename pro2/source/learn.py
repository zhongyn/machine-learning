import numpy as np
import featureVector as fvec

# Size of label 
groups = 0
with open("../data/newsgrouplabels.txt",'r') as f:
	for line in f:
		groups += 1
f.closed

# Size of vocabulary
voca = 0
with open("../data/vocabulary.txt",'r') as f:
	for line in f:
		voca += 1
f.closed

# Load training data and label
trainData = np.loadtxt("../data/train.data", delimiter=' ', dtype=int)
trainLabel = np.loadtxt("../data/train.label", delimiter=' ', dtype=int)


# Label list
labels = np.arange(1,groups+2)
# Num of docs for each label
numDocsY = np.histogram(trainLabel, bins=labels)[0]
# Learn p(y) using MLE for different labels
probY = np.histogram(trainLabel, bins=labels, density=True)[0]

# Bernoulli: Count the number of docs containing word i for each label
# Multinomial: Count the number of word i for each label
vectors = fvec.featureCount(trainData, trainLabel, groups, voca)
berVectors = vectors[0]
mulVectors = vectors[1]

numDocs = len(trainLabel)

# Learn Pi|y for i = 1,...,|V| using Laplace smoothing for both models
probIyBer = (berVectors+1)/(numDocsY[:,np.newaxis]+numDocs)
probIyMul = (mulVectors+1)/(np.sum(mulVectors,axis=1)[:,np.newaxis]+voca)

# Control the priors with Dirichlet distribution
alpRan = 6
alpha = [10**i for i in range(-5,1)]
probIyMulAlpha = np.zeros((alpRan,groups,voca))
for index,alp in enumerate(alpha):
	probIyMulAlpha[index] = (mulVectors+alp)/(np.sum(mulVectors,axis=1)[:,np.newaxis]+voca*alp)


# Save Pi|y for both models
params = np.array([groups,voca])
np.savez("learnBer.npz", probIyBer=probIyBer, probY=probY, params=params)
np.savez("learnMul.npz", probIyMul=probIyMul, probY=probY, params=params)
np.savez("learnMulAlp.npz", probIyMulAlpha=probIyMulAlpha, probY=probY, params=params)







	

	









































