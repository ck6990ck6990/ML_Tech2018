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
test_X, test_Y = loadData('testing_data.txt')
q = np.where(Y != 0)[0]
p = np.where(Y == 0)[0]
pp = np.where(test_Y != 0)[0]
qq = np.where(test_Y == 0)[0]
Y[q] = -1 
Y[p] = 1
test_Y[pp] = -1
test_Y[qq] = 1
logG = [0, 1, 2, 3, 4]
E = []
for i in logG:
    tmp = 10**i
    clf = svm.SVC(C=0.1, kernel='rbf', gamma=tmp)
    clf.fit(X, Y)
    predict_Y = clf.predict(test_X)
    E_out = cal_err(predict_Y, test_Y)
    print(E_out)
    E.append(E_out)
plt.plot(logG, E)
plt.show()