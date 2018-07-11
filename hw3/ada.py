import numpy as np
import matplotlib.pyplot as plt

iteration = 300

#read training data
mat = []
mat2 = []
weight = []
f = open('hw3_train.dat.txt')
lines = f.readlines()
for i in range(len(lines)):
    tmp = lines[i].split()
    for j in range(len(tmp)):
        tmp[j] = float(tmp[j])
    mat.append(tmp)
mat = np.array(mat)
features = mat[:, 0:2]
labels = mat[:, 2]
labels = np.array(labels)

#read testing data
f2 = open('hw3_test.dat.txt')
lines = f2.readlines()
for i in range(len(lines)):
    tmp = lines[i].split()
    for j in range(len(tmp)):
        tmp[j] = float(tmp[j])
    mat2.append(tmp)
mat2 = np.array(mat2)
features2 = mat2[:, 0:2]
labels2 = mat2[:, 2]
labels2 = np.array(labels2)

class Model:
    def __init__(self, size=iteration):
        self.model_weights = np.zeros((size))
        self.model_list = []

def cal_err(pred, label):
    s = (pred != label).sum()
    return s/len(pred)

def stump_classifier(features, feature_index, rule, threshold):
    K, n = features.shape
    results = np.ones((K))
    feature_values = features[:, feature_index]
    if rule == 'lt':
        results[np.where(feature_values <= threshold)[0]] = -1.0
    elif rule == 'gt':
        results[np.where(feature_values > threshold)[0]] = -1.0
        
    return results

def do_stump(features, labels, weight):
    K, n = features.shape
    min_err = 99999999
    stump = {}
    for i in range(n):
        pairs = [(i, j) for i, j in zip(features[:,i], labels)]
        pairs = np.sort(np.array(pairs, dtype=[('feature',float),('labels',float)]), order='feature')
        for j in range(K-1):
            for rule in ['lt', 'gt']:
                threshold = (pairs[j][0]+pairs[j+1][0])/2
                predict = stump_classifier(features, i, rule, threshold)
                is_error = (predict != labels)
                weighted_err = np.dot(weight.T, is_error)
                if weighted_err < min_err:
                    min_err = weighted_err
                    return_predict = predict.copy()
                    stump['feature_index'] = i
                    stump['threshold'] = threshold
                    stump['rule'] = rule
    return stump, return_predict, min_err

def adaboost_train(features, labels, iteration):
    K, n = features.shape
    weight = np.ones((K))/K
    m = Model(iteration)
    E_in_g = []
    E_in_G = []
    E_out_g = []
    E_out_G = []
    U_arr = []
    for i in range(iteration):
        U = np.sum(weight)
        stump, predict, weighted_err = do_stump(features, labels, weight)
        m.model_list.append(stump)
        m.model_weights[i] = 0.5 * np.log((1.0 - weighted_err) / max(weighted_err, 1e-16))
        weight = weight * np.exp(-1.0 * m.model_weights[i] * labels.T * predict)
        weight = weight / np.sum(weight)
        # err = cal_err(stump_classifier(features, stump['feature_index'], stump['rule'], stump['threshold']), labels)
        # E_in_g.append(err)
        # ERR = cal_err(adaboost_classify(features, m), labels) 
        # E_in_G.append(ERR)
        # err = cal_err(stump_classifier(features2, stump['feature_index'], stump['rule'], stump['threshold']), labels2)   
        # E_out_g.append(err)
        ERR = cal_err(adaboost_classify(features2, m), labels2)
        E_out_G.append(ERR)
        U_arr.append(U)
    # x = [i for i in range(1, 301)]
    # plt.plot(x, E_out_G)
    # plt.show()
    # print("E_in(g1) = ", E_in[0])
    # print("Alpha 1 = ", m.model_weights[0])
    # print(U_arr[-1])
    # print("E_out(g1) = ", E_out_g[0])
    # print('E_out(G)', E_out_G[-1])
    return m

def adaboost_classify(data, m):
    K, n = data.shape
    output = np.zeros((K))
    for i in range(len(m.model_list)):
        model_prediction = stump_classifier(data, m.model_list[i]['feature_index'], m.model_list[i]['rule'], m.model_list[i]['threshold'])
        output += m.model_weights[i] * model_prediction

    return np.sign(output)

m = adaboost_train(features, labels, iteration)
pred = adaboost_classify(features, m)






