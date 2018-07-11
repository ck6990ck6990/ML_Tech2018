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

X, Y = loadData('training_data.txt')
q = np.where(Y != 8)[0]
p = np.where(Y == 8)[0]
Y[q] = -1 
Y[p] = 1
logC = [-5, -3, -1, 1, 3]
w = []
num=[]
for i in logC:
	tmp = 10**i
	clf = svm.SVC(C=tmp, kernel='poly', degree=2, gamma=1, coef0=1)
	clf.fit(X, Y)
	predict_Y = clf.predict(X)
	print(clf.n_support_)
	tmp_n = clf.n_support_[0] + clf.n_support_[1]
	num.append(tmp_n)
	print('n_support_ = ', tmp_n)
plt.plot(logC, num)
plt.show()


