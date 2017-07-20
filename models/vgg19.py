import keras as k
from keras.models import Sequential
from keras.applications.vgg19 import VGG19
from keras.layers.core import Dense, Flatten
from keras.layers.normalization import BatchNormalization
from keras import optimizers


def vgg19(img_rows, img_cols, channels=3, num_classes=None):

    base_model = VGG19(include_top=False,
                       weights='imagenet',
                       input_shape=(img_rows,img_cols,channels))

    model = Sequential()
    # Batchnorm input
    model.add(BatchNormalization(input_shape=(img_rows,img_cols,channels)))
    # Base vgg19 model
    model.add(base_model)
    # Classifier
    model.add(Flatten())
    model.add(Dense(num_classes, activation='sigmoid'))

    #sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    #model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
    opt = optimizers.Adam(lr=0.001)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    return model
