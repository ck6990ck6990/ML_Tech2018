import numpy as np

def gaussian(x1, x2, gamma):
	diff = x1 - x2
	t = np.exp(-gamma*np.dot(diff, diff.T))
	return t

def sign(x):
	if x >= 0:
		return 1
	else:
		return -1

data = np.loadtxt('data.txt')
train_x = data[0:400, 0:10]
train_y = data[0:400, 10]
test_x = data[400:, 0:10]
test_y = data[400:, 10]
size = test_x.shape[0]
gamma_arr = [32, 2, 0.125]
lamda_arr = [0.001, 1, 1000]

for gamma in gamma_arr:
	for lamda in lamda_arr:
		# find beta
		kernel = np.zeros((400, 400))
		for i in range(400):
			for j in range(i+1):
				tmp = gaussian(train_x[i], train_x[j], gamma)
				kernel[i][j] = tmp
				kernel[j][i] = tmp
		beta = np.linalg.inv(lamda*np.eye(400) + kernel).dot(train_y)

		# calculate error
		ein = np.zeros(400)
		for i in range(400):
			for j in range(400):
				ein[i] = ein[i] + beta[j]*gaussian(train_x[i], train_x[j], gamma)
			ein[i] = sign(ein[i])
		E_in = sum(ein != train_y)/400

		eout = np.zeros(size)
		for i in range(size):
			for j in range(400):
				eout[i] = eout[i] + beta[j]*gaussian(test_x[i], train_x[j], gamma)
			eout[i] = sign(eout[i])
		E_out = sum(eout != test_y)/size
		print("gamma = ", gamma, " lambda = ", lamda)
		print("E_in = ", E_in, " E_out = ", E_out, "\n")




