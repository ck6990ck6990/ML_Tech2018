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
    q = np.where(Y != 0)[0]
    p = np.where(Y == 0)[0]
    Y[q] = -1 
    Y[p] = 1
    train = []
    for i in range(7291):
        train.append([ X[i], [Y[i]] ])
    return train
def cal_err(x, y):
    err = np.count_nonzero(x != y)/len(y)
    return err

train = loadData('training_data.txt')

logG = [-1, 0, 1, 2, 3]
vote = [0, 0, 0, 0, 0]
for times in range(100):
    np.random.shuffle(train)
    train_X = []
    train_Y = []
    test_X = []
    test_Y = []
    for i in range(1001, 7291, 1):
        train_X.append(train[i][0])
        train_Y.append(train[i][1][0])
    for i in range(1000):       
        test_X.append(train[i][0])
        test_Y.append(train[i][1][0])
    best_logG = 0
    best_E = 1
    E = []
    for i in logG:
        tmp = 10**i
        clf = svm.SVC(C=0.1, kernel='rbf', gamma=tmp)
        clf.fit(train_X, train_Y)
        predict_Y = clf.predict(test_X)
        E_val = cal_err(predict_Y, test_Y)
        E.append(E_val)
        if E_val < best_E:
            best_E = E_val
            best_logG = i
    # print(E)
    vote[best_logG+1] = vote[best_logG+1] + 1
print(vote)
plt.bar(logG, vote)
plt.show()



