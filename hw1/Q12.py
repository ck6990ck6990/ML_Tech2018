import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
def loadData(filename):
    data = pd.read_csv(filename, sep='  ', header=None, engine='python')
    data = data.as_matrix()
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y
def cal_err(x, y):
	err = np.count_nonzero(x != y)/len(y)
	return err

X, Y = loadData('training_data.txt')
q = np.where(Y != 8)[0]
p = np.where(Y == 8)[0]
Y[q] = -1 
Y[p] = 1
logC = [-5, -3, -1, 1, 3]
w = []
E=[]
for i in logC:
	tmp = 10**i
	clf = svm.SVC(C=tmp, kernel='poly', degree=2, gamma=1, coef0=1)
	clf.fit(X, Y)
	predict_Y = clf.predict(X)
	E_in = cal_err(predict_Y, Y)
	E.append(E_in)
	print('E_in = ', E_in)
	# print('w = ', clf.dual_coef_)
	# print('|w| = ', np.linalg.norm(clf.dual_coef_))
plt.plot(logC, E)
plt.show()

