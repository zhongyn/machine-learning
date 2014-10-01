
import numpy as np
from scipy import linalg

def linreg(x,y,lam):
	"""
	Solving linear regression with close-form analytical solution and regularization
	"""
	I = np.identity(x.shape[1])
	xt = x.T
	
	w = np.empty([len(lam), x.shape[1]])
	
	for index, item in enumerate(lam):
		w[index] = linalg.inv(item*I + xt.dot(x)).dot(xt).dot(y)

	return w

def sserr(x,y,w,lam):
	"""
	Sum-squared error with several lambdas
	"""
	sse = np.empty(len(lam))

	for index, item in enumerate(w):
		tem = x.dot(item)-y
		sse[index] = 0.5*tem.T.dot(tem) + 0.5*lam[index]*item.T.dot(item)

	return sse  

def sse(x,y,w,lam):
	"""
	Sum-squared error with one lambda
	"""
	tem = x.dot(w)-y
	sse = 0.5*tem.dot(tem) + 0.5*lam*w.dot(w)

	return sse

def test(x,y,w,lam):
	"""
	test error = abs(y-wx)
	"""
	error = np.empty([len(lam),len(y)])

	for index, item in enumerate(w):
		error[index] = abs(y - x.dot(item))

	return error