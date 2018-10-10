# !/usr/bin/env python
# -*- coding:utf-8 -*-

import keras
from keras import optimizers
from keras import regularizers
from keras import backend as K
from keras.losses import categorical_crossentropy as logloss
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.engine.topology import Input, Container
from keras.engine.training import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, advanced_activations, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Convolution2D, pooling, Lambda, concatenate
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model as keras_load_model
import numpy as np
import os

num_classes = 10
temperature = 20
# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model = keras_load_model('/home/ubuntu/knowledge-distillation/models/model.ep33.h5')

model.summary()

y_pred = model.predict(x_test)
acc = 0
for i in range(y_pred.shape[0]):
    if np.argmax(y_pred[i][:10]) == np.argmax(y_test):
        acc = acc + 1

print('Test accuracy:', acc / y_pred.shape[0])
