import numpy as np

def batch(x,y,epsilon):
	"""
	Batch perceptron algorithm
	"""
	row = len(x)
	col = x.shape[1]

	w = np.zeros(col)
	epoch = 0
	error = []

	while True:
		delta = np.zeros(col)
		perloss = 0.0
		for i in range(row):
			u = w.T.dot(x[i])
			if y[i]*u <= 0.0:
				perloss -= y[i]*u
				delta = delta - y[i]*x[i]
		delta = delta/row
		error.append(perloss)
		epoch += 1
		w = w - delta
		if np.linalg.norm(delta) < epsilon: break

	return [w,error]

			

def voted(x,y,times):
	"""
	Voted perceptron algorithm
	"""
	row = len(x)
	col = x.shape[1]

	w = np.zeros(col)
	v = [w]
	c = [0.0]
	error = []
	

	for t in range(times):
		perloss = 0.0
		for i in range(row):
			u = v[-1].T.dot(x[i])
			if y[i]*u <= 0.0:
				perloss -= y[i]*u
				w = w + y[i]*x[i]
				v.append(w)
				c.append(1)
			else: c[-1] += 1
		error.append(perloss)

	v = np.array(v)
	c = np.array(c)
	return [v,c,error]
	

def votepredict(x,w,c):
	"""
	Voted perceptron prediction
	"""
	row = len(x)
	y = np.empty(row)

	for r in range(row):
		y[r] = np.sign(c.dot(np.sign(w.dot(x[r]))))

	return y


























