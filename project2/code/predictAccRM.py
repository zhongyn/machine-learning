import numpy as np
import featureVector as fvec


predictMulFeatRemove = np.load("predictMulFeatRemove.npy")

testLabel = np.loadtxt("../data/test.label", delimiter=' ', dtype=int)

Acc = fvec.accPredict(testLabel,predictMulFeatRemove)
