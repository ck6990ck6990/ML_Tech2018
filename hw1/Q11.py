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
def cal(x):
	sum = 0
	for i in range(len(x)):
		sum = sum + x[i]*x[i]
	return math.sqrt(sum)

X, Y = loadData('training_data.txt')
q = np.where(Y != 0)[0]
p = np.where(Y == 0)[0]
Y[q] = -1 
Y[p] = 1
logC = [-5, -3, -1, 1, 3]
w = []
for i in logC:
	tmp = 10**i
	clf = svm.SVC(C=tmp, kernel='linear')
	clf.fit(X, Y)
	w.append(cal(clf.coef_[0]))
print(w)
plt.plot(logC, w)
plt.show()

