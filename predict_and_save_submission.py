import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import random
import cv2
import gc

import keras as k
from keras.models import load_model
from keras import optimizers

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import fbeta_score, matthews_corrcoef

from preprocess import process_image
from models.densenet169 import densenet169_model
from models.vgg19 import vgg19
from models.inceptionv4 import inception_v4_model


df_valid = pd.read_csv('data/valid.csv')

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


def f2_score(y_true, y_pred):
    y_true, y_pred, = np.array(y_true), np.array(y_pred)
    return fbeta_score(y_true, y_pred, beta=2, average='samples')


def find_f2score_threshold(p_valid, y_valid, try_all=False, verbose=False):
    best = 0
    best_score = -1
    totry = np.arange(0,1,0.005) if try_all is False else np.unique(p_valid)
    for t in totry:
        score = f2_score(y_valid, p_valid > t)
        if score > best_score:
            best_score = score
            best = t
    if verbose is True:
        print('Best score: ', round(best_score, 5), ' @ threshold =', best)
    return best


if __name__=='__main__':

    img_rows, img_cols = 224, 224 # Resolution of inputs
    channels = 3
    num_classes = 17
    test_weights_path = 'models/vgg19_weights.full.h5'
    #last_weights_path = 'models/inceptionv4_weights.last.h5'
    # Load our model
    #model = inception_v4_model(img_rows=img_rows, img_cols=img_cols, channels=channels, num_classes=num_classes)
    #model.load_weights(test_weights_path)
    model = vgg19(img_rows=img_rows, img_cols=img_cols, channels=channels, num_classes=num_classes)
    model.load_weights(test_weights_path)

    X_valid = []
    Y_valid = []
    for f, tags in tqdm(df_valid.values, miniters=1000):
        img = cv2.imread('data/valid/{}.jpg'.format(f))
        #img = process_image(img)
        img = cv2.resize(img, (224,224), interpolation=cv2.INTER_AREA)
        targets = np.zeros(17)
        for t in tags.split(' '):
            targets[label_map[t]] = 1
        X_valid.append(img)
        Y_valid.append(targets)

    X_valid = np.array(X_valid, np.float16) / 255.
    Y_valid = np.array(Y_valid, np.uint8)

    p_valid = model.predict(X_valid, batch_size=8)
    print(Y_valid)
    print(p_valid)
    print(fbeta_score(Y_valid, np.array(p_valid) > 0.2, beta=2, average='samples'))

    constant_threshold = find_f2score_threshold(p_valid, Y_valid, verbose=True)


    del X_valid, Y_valid

    df_test = pd.read_csv('submissions/sample_submission_v2.csv')
    X_test = []
    for f, tags in tqdm(df_test.values, miniters=1000):
        img = cv2.imread('data/test-jpg/{}.jpg'.format(f))
        #img = process_image(img)
        img = cv2.resize(img, (224,224), interpolation=cv2.INTER_AREA)
        X_test.append(img)

    X_test = np.array(X_test, np.float16) / 255.
    p_test = model.predict(X_test, batch_size=8)
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
    df_test.to_csv('submissions/submission_vgg19_full_data.csv', index=False)

    gc.collect()
