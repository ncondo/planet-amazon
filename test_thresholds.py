import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc

import keras as k
from keras.models import load_model
from models.base_model import base_model

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import fbeta_score, matthews_corrcoef

import cv2
from tqdm import tqdm

x_train = []
y_train = []

df_train = pd.read_csv('data/train_v2.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

labels = ['blow_down',
 'bare_ground',
 'conventional_mine',
 'blooming',
 'cultivation',
 'artisinal_mine',
 'haze',
 'primary',
 'slash_burn',
 'habitation',
 'clear',
 'road',
 'selective_logging',
 'partly_cloudy',
 'agriculture',
 'water',
 'cloudy']

label_map = {'agriculture': 14,
 'artisinal_mine': 5,
 'bare_ground': 1,
 'blooming': 3,
 'blow_down': 0,
 'clear': 10,
 'cloudy': 16,
 'conventional_mine': 2,
 'cultivation': 4,
 'habitation': 9,
 'haze': 6,
 'partly_cloudy': 13,
 'primary': 7,
 'road': 11,
 'selective_logging': 12,
 'slash_burn': 8,
 'water': 15}

for f, tags in tqdm(df_train.values, miniters=1000):
    img = cv2.imread('data/train-jpg/{}.jpg'.format(f))
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1
    x_train.append(cv2.resize(img, (128,128)))
    y_train.append(targets)

y_train = np.array(y_train, np.uint8)
x_train = np.array(x_train, np.float16) / 255.

print(x_train.shape)
print(y_train.shape)

trn_index = []
val_index = []
# change split value for getting different validation splits
split = .2
index = np.arange(len(x_train))
for i in tqdm(range(0,17)):
    sss = StratifiedShuffleSplit(n_splits=2, test_size=split, random_state=i)
    for train_index, test_index in sss.split(index,y_train[:,i]):
        X_train, X_test = index[train_index], index[test_index]
    # to ensure there is no repetetion within each split and between the splits
    trn_index = trn_index + list(set(list(X_train)) - set(trn_index) - set(val_index))
    val_index = val_index + list(set(list(X_test)) - set(val_index) - set(trn_index))

X_train = np.empty([32383,128,128,3])
Y_train = np.empty([32383,17])
for i in range(len(trn_index)):
    X_train[i] = x_train[trn_index[i]]
    Y_train[i] = y_train[trn_index[i]]

X_valid = np.empty([8096,128,128,3])
Y_valid = np.empty([8096,17])
for i in range(len(val_index)):
    X_valid[i] = x_train[val_index[i]]
    Y_valid[i] = y_train[val_index[i]]

del x_train
del y_train

print('X_train {}, Y_train {}'.format(len(X_train), len(Y_train)))
print('X_valid {}, Y_valid {}'.format(len(X_valid), len(Y_valid)))

model = load_model('models/base_model.h5')

p_valid = model.predict(X_valid, batch_size=32)
print(Y_valid)
print(p_valid)
print(fbeta_score(Y_valid, np.array(p_valid) > 0.2, beta=2, average='samples'))

out = np.array(p_valid)
threshold = np.arange(0.1, 0.9, 0.1)
acc = []
accuracies = []
best_threshold = np.zeros(out.shape[1])
for i in range(out.shape[1]):
    y_prob = np.array(out[:,i])
    for j in threshold:
        y_pred = [1 if prob >= j else 0 for prob in y_prob]
        acc.append( matthews_corrcoef(Y_valid[:,i], y_pred))
    acc = np.array(acc)
    index = np.where(acc==acc.max())
    accuracies.append(acc.max())
    best_threshold[i] = threshold[index[0][0]]
    acc = []

print(best_threshold)

y_pred = np.array([[1 if out[i,j]>=best_threshold[j] else 0 for j in range(Y_valid.shape[1])] for i in range(len(Y_valid))])
print(fbeta_score(Y_valid, y_pred, beta=2, average='samples'))

del X_train, X_valid
gc.collect()

df_test = pd.read_csv('data/sample_submission_v2.csv')
X_test = []
for f, tags in tqdm(df_test.values, miniters=1000):
    img = cv2.imread('data/test-jpg/{}.jpg'.format(f))
    X_test.append(cv2.resize(img, (128,128)))

X_test = np.array(X_test, np.float16) / 255.
p_test = model.predict(X_test, batch_size=32)
out = np.array(p_test)

y_pred = np.array([[1 if out[i,j]>=best_threshold[j] else 0 for j in range(p_test.shape[1])] for i in range(len(p_test))])
result = pd.DataFrame(y_pred, columns=labels)
preds = []
for i in tqdm(range(result.shape[0]), miniters=1000):
    a = result.ix[[i]]
    a = a.apply(lambda x: x == 1, axis=1)
    a = a.transpose()
    a = a.loc[a[i] == True]
    ' '.join(list(a.index))
    preds.append(' '.join(list(a.index)))

df_test['tags'] = preds
df_test.to_csv('submissions/submission_base_model_thresh_128.csv', index=False)

gc.collect()
k.clear_session()
