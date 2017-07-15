import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import cv2
from tqdm import tqdm

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import fbeta_score

import keras as k
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from models.base_model import base_model


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


best_weights_path = 'models/weights.best.h5'
model = base_model(128,128,3)
epochs_arr = [20, 5, 5]
learn_rates = [0.001, 0.0001, 0.00001]
for learn_rate, epochs in zip(learn_rates, epochs_arr):
    opt = optimizers.Adam(lr=learn_rate)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=0),
                 ModelCheckpoint(best_weights_path, monitor='val_loss',
                 save_best_only=True, verbose=0)]

    model.fit(x=X_train, y=Y_train, validation_data=(X_valid, Y_valid),
              batch_size=32, epochs=epochs, verbose=2, callbacks=callbacks, shuffle=True)

model.load_weights('models/weights.best.h5')
model.save('models/base_model.h5')
del model

model = load_model('models/base_model.h5')

p_valid = model.predict(X_valid, batch_size=32)
print(Y_valid)
print(p_valid)
print(fbeta_score(Y_valid, np.array(p_valid) > 0.2, beta=2, average='samples'))

del X_train
del X_valid

df_test = pd.read_csv('data/sample_submission_v2.csv')
X_test = []
for f, tags in tqdm(df_test.values, miniters=1000):
    img = cv2.imread('data/test-jpg/{}.jpg'.format(f))
    X_test.append(cv2.resize(img, (128,128)))

X_test = np.array(X_test, np.float16) / 255.
p_test = model.predict(X_test, batch_size=32)
result = pd.DataFrame(p_test, columns=labels)
preds = []
for i in tqdm(range(result.shape[0]), miniters=1000):
    a = result.ix[[i]]
    a = a.apply(lambda x: x > 0.2, axis=1)
    a = a.transpose()
    a = a.loc[a[i] == True]
    ' '.join(list(a.index))
    preds.append(' '.join(list(a.index)))

df_test['tags'] = preds
df_test.to_csv('submissions/submission_base_model_128.csv', index=False)
gc.collect()
