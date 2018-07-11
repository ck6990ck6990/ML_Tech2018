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
def kernel(a, b):
    sum = 0
    for i in range(a.shape[0]):
        sum = sum + (a[i]-b[i])**2
    t = sum
    return math.exp(-1*80*t)
    
def cal_w(v, x):
    sum = 0
    for i in range(v.shape[0]):
        for j in range(v.shape[0]):
            sum = sum + v[i]*v[j]*kernel(x[i], x[j])
    return math.sqrt(sum)

X, Y = loadData('training_data.txt')
q = np.where(Y != 0)[0]
p = np.where(Y == 0)[0]
Y[q] = -1 
Y[p] = 1
logC = [-3, -2, -1, 0, 1]
d = []
for i in logC:
    tmp = 10**i
    clf = svm.SVC(C=tmp, kernel='rbf', gamma=80)
    clf.fit(X, Y)
    dist = 1/cal_w(clf.dual_coef_[0], clf.support_vectors_)
    d.append(dist)
    print("dist = ", dist)
plt.plot(logC, d)
plt.show()
