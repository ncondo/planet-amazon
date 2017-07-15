import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from shutil import copyfile
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit


df_train = pd.read_csv('data/train_v2.csv')

# mapping labels to integer classes
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

y_train = []
# labels for the train dataset
for f, tags in tqdm(df_train.values, miniters=1000):
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1
    y_train.append(targets)

y_train = np.array(y_train, np.uint8)

trn_index = []
val_index = []
# change split value for getting different validation splits
split = .2
index = np.arange(len(df_train))
for i in tqdm(range(0,17)):
    sss = StratifiedShuffleSplit(n_splits=2, test_size=split, random_state=8)
    for train_index, test_index in sss.split(index,y_train[:,i]):
        X_train, X_test = index[train_index], index[test_index]
    # to ensure there is no repetetion within each split and between the splits
    trn_index = trn_index + list(set(list(X_train)) - set(trn_index) - set(val_index))
    val_index = val_index + list(set(list(X_test)) - set(val_index) - set(trn_index))

print(len(trn_index), len(val_index))

os.mkdir('data/train/')
os.mkdir('data/valid/')
train = pd.DataFrame()
for i in range(len(trn_index)):
    df = df_train.iloc[[trn_index[i]]]
    f = str(df['image_name'].item())
    copyfile('data/train-jpg/{}.jpg'.format(f), 'data/train/{}.jpg'.format(f))
    train = train.append(df, ignore_index=True)

valid = pd.DataFrame()
for i in range(len(val_index)):
    df = df_train.iloc[[val_index[i]]]
    f = str(df['image_name'].item())
    copyfile('data/train-jpg/{}.jpg'.format(f), 'data/valid/{}.jpg'.format(f))
    valid = valid.append(df, ignore_index=True)

train.to_csv('data/train.csv', index=False)
valid.to_csv('data/valid.csv', index=False)
