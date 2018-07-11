from surprise import KNNBasic, SVD, NMF, SVDpp, BaselineOnly, KNNWithMeans
from surprise import Dataset, Reader
from surprise import evaluate, NormalPredictor
from surprise.model_selection import cross_validate
import pandas as pd
import numpy as np
import csv

path = '2018_spring/MLT/final/data/book_ratings_train.csv' # 260203 rows
path2 = '2018_spring/MLT/final/data/book_ratings_test.csv'

f = open('KNNWithMeans.csv', 'w')
data = np.array(pd.read_csv(path))
userID = data[:,0]
bookID = data[:,1]
rating = data[:,2]
dic = {'userID':userID, 'bookID':bookID, 'rating':rating}
df = pd.DataFrame(dic)
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(df[['userID', 'bookID', 'rating']], reader)
training_data = data.build_full_trainset()
# print(training_data[0])
# data.split(n_folds=3)
print('Build Finished')
print("Start fit")
# print(uid)

algo = KNNWithMeans()
algo.fit(training_data)

print('Fit Done')

data2 = np.array(pd.read_csv(path2))
uid = data2[:,0]
bid = data2[:,1]
test_len = len(uid)
for i in range(test_len):
	pred = algo.predict(uid[i], bid[i])
	print("uid = %s, bid = %s, set = %f" % (uid[i], bid[i], pred.est))
	f.write(str(int(round(pred.est)))+'\n')
	# est.append(pred.est)
# print(est)
f.close()
