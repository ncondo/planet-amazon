import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import random
import cv2
import gc

import keras as k
from keras.models import load_model
from keras import optimizers

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

    get_fscore = True
    find_thresholds = False
    create_submission = True
    img_rows, img_cols = 224, 224 # Resolution of inputs
    channels = 3
    num_classes = 17
    # Define vgg weights paths
    vgg_best_weights_path = 'models/vgg19_weights.best.h5'
    vgg_last_weights_path = 'models/vgg19_weights.last.h5'
    vgg_test_weights_path = 'models/vgg19_weights.test.h5'
    vgg_full_weights_path = 'models/vgg19_weights.full.h5'
    # Define densenet weights paths
    densenet_best_weights_path = 'models/densenet169_weights.best.h5'
    densenet_last_weights_path = 'models/densenet169_weights.last.h5'
    densenet_test_weights_path = 'models/densenet169_weights.test.h5'
    densenet_full_weights_path = 'models/densenet169_weights.full.h5'
    # Define inception weights paths
    inception_best_weights_path = 'models/inceptionv4_weights.best.h5'
    inception_last_weights_path = 'models/inceptionv4_weights.last.h5'
    inception_test_weights_path = 'models/inceptionv4_weights.test.h5'
    inception_full_weights_path = 'models/inceptionv4_weights.full.h5'
    # Load vgg models
    vgg_best = vgg19(img_rows=img_rows, img_cols=img_cols, channels=channels, num_classes=num_classes)
    vgg_best.load_weights(vgg_best_weights_path)
    vgg_last = vgg19(img_rows=img_rows, img_cols=img_cols, channels=channels, num_classes=num_classes)
    vgg_last.load_weights(vgg_last_weights_path)
    vgg_test = vgg19(img_rows=img_rows, img_cols=img_cols, channels=channels, num_classes=num_classes)
    vgg_test.load_weights(vgg_test_weights_path)
    vgg_full = vgg19(img_rows=img_rows, img_cols=img_cols, channels=channels, num_classes=num_classes)
    vgg_full.load_weights(vgg_full_weights_path)
    # Load densenet models
    densenet_best = densenet169_model(img_rows=img_rows, img_cols=img_cols, channels=channels, num_classes=num_classes)
    densenet_best.load_weights(densenet_best_weights_path)
    densenet_last = densenet169_model(img_rows=img_rows, img_cols=img_cols, channels=channels, num_classes=num_classes)
    densenet_last.load_weights(densenet_last_weights_path)
    densenet_test = densenet169_model(img_rows=img_rows, img_cols=img_cols, channels=channels, num_classes=num_classes)
    densenet_test.load_weights(densenet_test_weights_path)
    densenet_full = densenet169_model(img_rows=img_rows, img_cols=img_cols, channels=channels, num_classes=num_classes)
    densenet_full.load_weights(densenet_full_weights_path)
    # Load inception models
    inception_best = inception_v4_model(img_rows=299, img_cols=299, channels=channels, num_classes=num_classes, dropout_keep_prob=0.2)
    inception_best.load_weights(inception_best_weights_path)
    inception_last = inception_v4_model(img_rows=299, img_cols=299, channels=channels, num_classes=num_classes, dropout_keep_prob=0.2)
    inception_last.load_weights(inception_last_weights_path)
    inception_test = inception_v4_model(img_rows=299, img_cols=299, channels=channels, num_classes=num_classes, dropout_keep_prob=0.2)
    inception_test.load_weights(inception_test_weights_path)
    inception_full = inception_v4_model(img_rows=299, img_cols=299, channels=channels, num_classes=num_classes, dropout_keep_prob=0.2)
    inception_full.load_weights(inception_full_weights_path)

    # Check f2 score on validation set
    if get_fscore:
        Y_valid = []
        p_valid = np.zeros(17)
        for f, tags in tqdm(df_valid.values, miniters=1000):
            img = cv2.imread('data/valid/{}.jpg'.format(f))
            #img = process_image(img)
            # Get 224x224 image for densenet and vgg models
            img_224 = cv2.resize(img, (224,224), interpolation=cv2.INTER_AREA)
            img_224 = np.array(img_224, dtype=np.float16) / 255.
            img_224 = np.expand_dims(img_224, axis=0)
            # Get 299x299 image for inception models
            img_299 = cv2.resize(img, (299,299), interpolation=cv2.INTER_AREA)
            img_299 = np.array(img_299, dtype=np.float16) / 255.
            img_299 = np.expand_dims(img_299, axis=0)
            # Get predictions from vgg models
            vgg_best_pred = vgg_best.predict(img_224, batch_size=1)
            vgg_last_pred = vgg_last.predict(img_224, batch_size=1)
            vgg_test_pred = vgg_test.predict(img_224, batch_size=1)
            vgg_full_pred = vgg_full.predict(img_224, batch_size=1)
            # Get predictions from densenet models
            densenet_best_pred = densenet_best.predict(img_224, batch_size=1)
            densenet_last_pred = densenet_last.predict(img_224, batch_size=1)
            densenet_test_pred = densenet_test.predict(img_224, batch_size=1)
            densenet_full_pred = densenet_full.predict(img_224, batch_size=1)
            # Get predictions from inception models
            inception_best_pred = inception_best.predict(img_299, batch_size=1)
            inception_last_pred = inception_last.predict(img_299, batch_size=1)
            inception_test_pred = inception_test.predict(img_299, batch_size=1)
            inception_full_pred = inception_full.predict(img_299, batch_size=1)
            # Combine predictions and get average
            total = np.add(vgg_best_pred, vgg_last_pred)
            total = np.add(total, vgg_test_pred)
            total = np.add(total, vgg_full_pred)
            total = np.add(total, densenet_best_pred)
            total = np.add(total, densenet_last_pred)
            total = np.add(total, densenet_test_pred)
            total = np.add(total, densenet_full_pred)
            total = np.add(total, inception_best_pred)
            total = np.add(total, inception_last_pred)
            total = np.add(total, inception_test_pred)
            total = np.add(total, inception_full_pred)
            avg = total / 12
            p_valid = np.vstack((p_valid, avg))
            targets = np.zeros(17)
            for t in tags.split(' '):
                targets[label_map[t]] = 1
            Y_valid.append(targets)

        Y_valid = np.array(Y_valid, np.uint8)
        p_valid = np.delete(p_valid, 0, axis=0)

        print(Y_valid)
        print(p_valid)
        print(fbeta_score(Y_valid, np.array(p_valid) > 0.2, beta=2, average='samples'))

    constant_threshold = find_f2score_threshold(p_valid, Y_valid, verbose=True)

    # Find different thresholds for each label
    if find_thresholds:
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


    # Create submission
    if create_submission:
        del p_valid, Y_valid
        df_test = pd.read_csv('submissions/sample_submission_v2.csv')
        p_test = np.zeros(17)
        for f, tags in tqdm(df_test.values, miniters=1000):
            img = cv2.imread('data/test-jpg/{}.jpg'.format(f))
            #img = process_image(img)
            # Get 224x224 image for densenet and vgg models
            img_224 = cv2.resize(img, (224,224), interpolation=cv2.INTER_AREA)
            img_224 = np.array(img_224, dtype=np.float16) / 255.
            img_224 = np.expand_dims(img_224, axis=0)
            # Get 299x299 image for inception models
            img_299 = cv2.resize(img, (299,299), interpolation=cv2.INTER_AREA)
            img_299 = np.array(img_299, dtype=np.float16) / 255.
            img_299 = np.expand_dims(img_299, axis=0)
            # Get predictions from vgg models
            vgg_best_pred = vgg_best.predict(img_224, batch_size=1)
            vgg_last_pred = vgg_last.predict(img_224, batch_size=1)
            vgg_test_pred = vgg_test.predict(img_224, batch_size=1)
            vgg_full_pred = vgg_full.predict(img_224, batch_size=1)
            # Get predictions from densenet models
            densenet_best_pred = densenet_best.predict(img_224, batch_size=1)
            densenet_last_pred = densenet_last.predict(img_224, batch_size=1)
            densenet_test_pred = densenet_test.predict(img_224, batch_size=1)
            densenet_full_pred = densenet_full.predict(img_224, batch_size=1)
            # Get predictions from inception models
            inception_best_pred = inception_best.predict(img_299, batch_size=1)
            inception_last_pred = inception_last.predict(img_299, batch_size=1)
            inception_test_pred = inception_test.predict(img_299, batch_size=1)
            inception_full_pred = inception_full.predict(img_299, batch_size=1)
            # Combine predictions and get average
            total = np.add(vgg_best_pred, vgg_last_pred)
            total = np.add(total, vgg_test_pred)
            total = np.add(total, vgg_full_pred)
            total = np.add(total, densenet_best_pred)
            total = np.add(total, densenet_last_pred)
            total = np.add(total, densenet_test_pred)
            total = np.add(total, densenet_full_pred)
            total = np.add(total, inception_best_pred)
            total = np.add(total, inception_last_pred)
            total = np.add(total, inception_test_pred)
            total = np.add(total, inception_full_pred)
            avg = total / 12
            p_test = np.vstack((p_test, avg))

        p_test = np.delete(p_test, 0, axis=0)
        #y_test = np.array([[1 if out[i,j]>=best_threshold[j] else 0 for j in range(p_test.shape[1])] for i in range(len(p_test))])
        result = pd.DataFrame(p_test, columns=labels)
        preds = []
        for i in tqdm(range(result.shape[0]), miniters=1000):
            a = result.ix[[i]]
            a = a.apply(lambda x: x >= constant_threshold, axis=1)
            a = a.transpose()
            a = a.loc[a[i] == True]
            ' '.join(list(a.index))
            preds.append(' '.join(list(a.index)))

        df_test['tags'] = preds
        df_test.to_csv('submissions/submission_ensemble_thresh_7-20_2.csv', index=False)

    gc.collect()
