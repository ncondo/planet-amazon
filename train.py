import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import random
import gc
import cv2
import sys
sys.path.append('models/custom_layers')

import keras as k
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

from sklearn.metrics import fbeta_score

from models.base_model import base_model
from models.inceptionv4 import inception_v4_model
from models.vgg19 import vgg19
from models import densenet169
from models.custom_layers import scale_layer

from preprocess import process_image


df_train = pd.read_csv('data/train_v2.csv')
#df_valid = pd.read_csv('data/valid.csv')

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


def train_generator(batch_size):
    while True:
        x_train = []
        y_train = []
        for i in range(batch_size):
            rand_index = random.randrange(len(df_train))
            df = df_train.iloc[[rand_index]]
            f = str(df['image_name'].item())
            tags = df['tags'].item()
            img = cv2.imread('data/train-jpg/{}.jpg'.format(f))
            img = process_image(img)
            targets = np.zeros(17)
            for t in tags.split(' '):
                targets[label_map[t]] = 1
            x_train.append(img)
            y_train.append(targets)
        x_train = np.array(x_train, np.float16) / 255.
        y_train = np.array(y_train, np.uint8)
        yield x_train, y_train


def valid_generator(batch_size):
    while True:
        x_valid = []
        y_valid = []
        for i in range(batch_size):
            rand_index = random.randrange(len(df_valid))
            df = df_valid.iloc[[rand_index]]
            f = str(df['image_name'].item())
            tags = df['tags'].item()
            img = cv2.imread('data/valid/{}.jpg'.format(f))
            img = cv2.resize(img, (224,224), interpolation=cv2.INTER_AREA)
            targets = np.zeros(17)
            for t in tags.split(' '):
                targets[label_map[t]] = 1
            x_valid.append(img)
            y_valid.append(targets)
        x_valid = np.array(x_valid, np.float16) / 255.
        y_valid = np.array(y_valid, np.uint8)
        yield x_valid, y_valid


if __name__=='__main__':

    #best_weights_path = 'models/densenet169_full_data_weights.best.h5'
    last_weights_path = 'models/inceptionv4_weights.full.h5'
    img_rows, img_cols = 299, 299 # Resolution of inputs
    channels = 3
    num_classes = 17
    batch_size = 8
    # Load our model
    #model = base_model(128,128,3)
    #model = vgg19(img_rows=img_rows, img_cols=img_cols, channels=channels, num_classes=num_classes)
    #model.load_weights('models/vgg19_weights.best.h5')
    model = inception_v4_model(img_rows=img_rows, img_cols=img_cols, channels=channels, num_classes=num_classes, dropout_keep_prob=0.2)
    model.load_weights('models/inceptionv4_weights.best.h5')
    epochs_arr = [2]
    learn_rates = [0.00001]
    for learn_rate, epochs in zip(learn_rates, epochs_arr):
        opt = optimizers.Adam(lr=learn_rate)
        model.compile(loss='binary_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
        """
        callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=0),
                     ModelCheckpoint(best_weights_path, monitor='val_loss',
                     save_best_only=True, verbose=0)]
        """

        model.fit_generator(train_generator(batch_size=batch_size),
                            samples_per_epoch=40480 // batch_size,
                            epochs=epochs,
                            validation_data=None,#valid_generator(batch_size=batch_size),
                            validation_steps=None,#4048 // batch_size,
                            #callbacks=callbacks,
                            verbose=2)

    model.save_weights(last_weights_path)
