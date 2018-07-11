import numpy as np

def linear(x1, x2):
	t = np.dot(x1, x2.T)
	return t

def sign(x):
	if x >= 0:
		return 1
	else:
		return -1

data = np.loadtxt('data.txt')
train_x = data[0:400, 0:10]
train_x = np.insert(train_x, 0, 1, axis=1)
train_y = data[0:400, 10]
test_x = data[400:, 0:10]
test_x = np.insert(test_x, 0, 1, axis=1)
test_y = data[400:, 10]
size = test_x.shape[0]
lamda_arr = [0.01, 0.1, 1, 10, 100]

for lamda in lamda_arr:
	# find beta
	kernel = np.zeros((400, 400))
	for i in range(400):
		for j in range(i+1):
			tmp = linear(train_x[i], train_x[j])
			kernel[i][j] = tmp
			kernel[j][i] = tmp
	beta = np.linalg.inv(lamda*np.eye(400) + kernel).dot(train_y)

	# calculate error
	ein = np.zeros(400)
	for i in range(400):
		for j in range(400):
			ein[i] = ein[i] + beta[j]*linear(train_x[i], train_x[j])
		ein[i] = sign(ein[i])
	E_in = sum(ein != train_y)/400

	eout = np.zeros(size)
	for i in range(size):
		for j in range(400):
			eout[i] = eout[i] + beta[j]*linear(test_x[i], train_x[j])
		eout[i] = sign(eout[i])
	E_out = sum(eout != test_y)/size
	print("lambda = ", lamda)
	print("E_in = ", E_in, " E_out = ", E_out, "\n")




