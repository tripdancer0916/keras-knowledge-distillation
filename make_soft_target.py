# !/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np

import keras
from keras import optimizers
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
import os

model = keras_load_model('teachermodel.ep11.h5')

batch_size = 100
num_classes = 10
epochs = 300
num_predictions = 20


def softmax(a):
    # 一番大きい値を取得
    c = np.max(a)
    # 各要素から一番大きな値を引く（オーバーフロー対策）
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    # 要素の値/全体の要素の合計
    y = exp_a / sum_exp_a

    return y


# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

z = model.predict(x_train[:100])
print(z[0])

model.layers.pop()
model = Model(model.input, model.layers[-1].output)

logits = model.predict(x_train[:100])
print(logits[0])
print(softmax(logits[0]))
