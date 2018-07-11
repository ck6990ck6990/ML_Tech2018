import numpy as np
from random import randint

def linear(x1, x2):
	t = np.dot(x1, x2.T)
	return t

def sign(x):
	if x >= 0:
		return 1
	else:
		return -1

data = np.loadtxt('data.txt')

lamda_arr = [0.01, 0.1, 1, 10, 100]
for lamda in lamda_arr:
	total_E_in = 0
	total_E_out = 0	
	for iteration in range(250):
		rand = []
		train_x = []
		train_y = []
		for i in range(400):
			rand.append(randint(0, 399))
			train_x.append(data[rand[i], 0:10])
			train_y.append(data[rand[i], 10])
		train_x = np.array(train_x)
		train_y = np.array(train_y)
		train_x = np.insert(train_x, 0, 1, axis=1)

		test_x = []
		test_y = []
		for i in range(400):
			if i not in rand:
				test_x.append(data[i, 0:10])
				test_y.append(data[i, 10])
		test_x = np.array(test_x)
		test_y = np.array(test_y)

		test_x = np.insert(test_x, 0, 1, axis=1)

		size = test_x.shape[0]

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
		total_E_in = total_E_in + E_in
		
		eout = np.zeros(size)
		for i in range(size):
			for j in range(400):
				eout[i] = eout[i] + beta[j]*linear(test_x[i], train_x[j])
			eout[i] = sign(eout[i])
		E_out = sum(eout != test_y)/size
		total_E_out = total_E_out + E_out
	print("lambda = ", lamda)
	print("E_in = ", total_E_in/250, " E_out = ", total_E_out/250, "\n")




