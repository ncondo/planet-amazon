# -*- coding: utf-8 -*-
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import cv2
from tqdm import tqdm

from keras.layers import Input, merge, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
import keras.backend as K

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import fbeta_score

from custom_layers.scale_layer import Scale


def densenet169_model(img_rows, img_cols, color_type=1, nb_dense_block=4, growth_rate=32, nb_filter=64, reduction=0.5, dropout_rate=0.0, weight_decay=1e-4, num_classes=None):
    '''
    DenseNet 169 Model for Keras
    Model Schema is based on
    https://github.com/flyyufelix/DenseNet-Keras
    ImageNet Pretrained Weights
    Theano: https://drive.google.com/open?id=0Byy2AcGyEVxfN0d3T1F1MXg0NlU
    TensorFlow: https://drive.google.com/open?id=0Byy2AcGyEVxfSEc5UC1ROUFJdmM
    # Arguments
        nb_dense_block: number of dense blocks to add to end
        growth_rate: number of filters to add per dense block
        nb_filter: initial number of filters
        reduction: reduction factor of transition blocks.
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        classes: optional number of classes to classify images
        weights_path: path to pre-trained weights
    # Returns
        A Keras model instance.
    '''
    eps = 1.1e-5

    # compute compression factor
    compression = 1.0 - reduction

    # Handle Dimension Ordering for different backends
    global concat_axis
    if K.image_dim_ordering() == 'tf':
      concat_axis = 3
      img_input = Input(shape=(128, 128, 3), name='data')
    else:
      concat_axis = 1
      img_input = Input(shape=(3, 128, 128), name='data')

    # From architecture for ImageNet (Table 1 in the paper)
    nb_filter = 64
    nb_layers = [6,12,32,32] # For DenseNet-169

    # Initial convolution
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Convolution2D(nb_filter, 7, 7, subsample=(2, 2), name='conv1', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x)
    x = Scale(axis=concat_axis, name='conv1_scale')(x)
    x = Activation('relu', name='relu1')(x)
    x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx+2
        x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

        # Add transition_block
        x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    final_stage = stage + 1
    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv'+str(final_stage)+'_blk_bn')(x)
    x = Scale(axis=concat_axis, name='conv'+str(final_stage)+'_blk_scale')(x)
    x = Activation('relu', name='relu'+str(final_stage)+'_blk')(x)

    x_fc = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)
    x_fc = Dense(1000, name='fc6')(x_fc)
    x_fc = Activation('softmax', name='prob')(x_fc)

    model = Model(img_input, x_fc, name='densenet')

    if K.image_dim_ordering() == 'th':
      # Use pre-trained weights for Theano backend
      weights_path = 'imagenet_models/densenet169_weights_th.h5'
    else:
      # Use pre-trained weights for Tensorflow backend
      weights_path = 'imagenet_models/densenet169_weights_tf.h5'

    model.load_weights(weights_path, by_name=True)

    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    x_newfc = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)
    x_newfc = Dense(num_classes, name='fc6')(x_newfc)
    x_newfc = Activation('softmax', name='prob')(x_newfc)

    model = Model(img_input, x_newfc)

    # Learning rate is changed to 0.001
    #sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    #model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def conv_block(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
    '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
        # Arguments
            x: input tensor
            stage: index for dense block
            branch: layer index within each dense block
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x1_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x1_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x1')(x)
    x = Convolution2D(inter_channel, 1, 1, name=conv_name_base+'_x1', bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x2_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x2_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x2')(x)
    x = ZeroPadding2D((1, 1), name=conv_name_base+'_x2_zeropadding')(x)
    x = Convolution2D(nb_filter, 3, 3, name=conv_name_base+'_x2', bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_filter: number of filters
            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''

    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_scale')(x)
    x = Activation('relu', name=relu_name_base)(x)
    x = Convolution2D(int(nb_filter * compression), 1, 1, name=conv_name_base, bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

    return x


def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
    '''

    eps = 1.1e-5
    concat_feat = x

    for i in range(nb_layers):
        branch = i+1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
        concat_feat = merge([concat_feat, x], mode='concat', concat_axis=concat_axis, name='concat_'+str(stage)+'_'+str(branch))

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter


if __name__ == '__main__':

    x_train = []
    y_train = []

    df_train = pd.read_csv('../data/train_v2.csv')

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
        img = cv2.imread('../data/train-jpg/{}.jpg'.format(f))
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


    best_weights_path = 'weights.best.h5'
    img_rows, img_cols = 128, 128 # Resolution of inputs
    channel = 3
    num_classes = 17
    batch_size = 32
    # Load our model
    model = densenet169_model(img_rows=img_rows, img_cols=img_cols, color_type=channel, num_classes=num_classes)
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
                  batch_size=batch_size, epochs=epochs, verbose=2, callbacks=callbacks, shuffle=True)

    model.load_weights('weights.best.h5')
    model.save('densenet169_model.h5')
    del model

    model = load_model('densenet169_model.h5')

    p_valid = model.predict(X_valid, batch_size=32)
    print(Y_valid)
    print(p_valid)
    print(fbeta_score(Y_valid, np.array(p_valid) > 0.2, beta=2, average='samples'))

    del X_train
    del X_valid

    df_test = pd.read_csv('../data/sample_submission_v2.csv')
    X_test = []
    for f, tags in tqdm(df_test.values, miniters=1000):
        img = cv2.imread('../data/test-jpg/{}.jpg'.format(f))
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
    df_test.to_csv('../submissions/submission_densenet169_model_224.csv', index=False)
    gc.collect()
