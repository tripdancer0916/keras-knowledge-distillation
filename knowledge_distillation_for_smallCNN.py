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
from keras.callbacks import Callback
import pytz
from datetime import datetime
import os

batch_size = 100
num_classes = 10
epochs = 300
num_predictions = 20

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

teacher_model = keras_load_model('teacher-model.ep55.h5')

teacher_model.layers.pop()
model = Model(teacher_model.input, teacher_model.layers[-1].output)

logits_train = model.predict(x_train)
y_train_ = np.concatenate([y_train, logits_train], axis=1)

logits_test = model.predict(x_test)
y_test_ = np.concatenate([y_test, logits_test], axis=1)

print('finish preparing.')

temperature = 20


def knowledge_distillation_loss(y_true, y_pred, lambda_const):
    # split in
    #    onehot hard true targets
    #    logits from xception
    y_true, logits = y_true[:, :num_classes], y_true[:, num_classes:]

    # convert logits to soft targets
    y_soft = K.softmax(logits / temperature)

    # split in
    #    usual output probabilities
    #    probabilities made softer with temperature
    y_pred, y_pred_soft = y_pred[:, :num_classes], y_pred[:, num_classes:]

    return (1-lambda_const) * logloss(y_true, y_pred) + \
           lambda_const * temperature * temperature * logloss(y_soft, y_pred_soft)


def accuracy(y_true, y_pred):
    y_true = y_true[:, :num_classes]
    y_pred = y_pred[:, :num_classes]
    return categorical_accuracy(y_true, y_pred)


def categorical_crossentropy(y_true, y_pred):
    y_true = y_true[:, :num_classes]
    y_pred = y_pred[:, :num_classes]
    return logloss(y_true, y_pred)


# logloss with only soft probabilities and targets
def soft_logloss(y_true, y_pred):
    logits = y_true[:, num_classes:]
    y_soft = K.softmax(logits/temperature)
    y_pred_soft = y_pred[:, num_classes:]
    return logloss(y_soft, y_pred_soft)


class TrainingCallback(Callback):
    def __init__(self, model, model_prefix):
        super(TrainingCallback, self).__init__()
        self.model = model
        self.model_prefix = model_prefix

    def on_epoch_end(self, epoch, logs={}):
        acc = model.evaluate()
        print('val_acc: ', acc)
        time_stamp = datetime.strftime(datetime.now(pytz.timezone('Japan')), '%m%d%H%M')
        model_name = '{}_{}_epoch_{:03d}_ACC_{:.4f}_loss_{:.4f}.aiinside'.format(
            time_stamp, self.model_prefix, epoch + 1, acc, logs['loss'])
        save_model_path = os.path.join('./', model_name)

        self.model.train_model.save(save_model_path)


class StudentModel(object):
    def __init__(self):
        self.train_model = None
        self.temperature = 5.0
        self.train_model = self.prepare()

    def prepare(self):
        input_layer = Input(x_train.shape[1:])
        x = Convolution2D(32, (3, 3), padding='same')(input_layer)
        x = BatchNormalization()(x)
        x = advanced_activations.LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        x = Dropout(0.5)(x)
        x = Convolution2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = advanced_activations.LeakyReLU(alpha=0.1)(x)
        x = Convolution2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = advanced_activations.LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        x = Dropout(0.5)(x)
        x = Convolution2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = advanced_activations.LeakyReLU(alpha=0.1)(x)

        x = pooling.GlobalAveragePooling2D()(x)
        logits = Dense(10, activation=None)(x)
        probabilities = Activation('softmax')(logits)

        logits_T = Lambda(lambda x: x/temperature)(logits)
        probabilities_T = Activation('softmax')(logits_T)

        output = concatenate([probabilities, probabilities_T])

        model = Model(input_layer, output)
        return model

    def evaluate(self):
        y_pred = self.train_model.predict(x_test)
        acc = 0
        for i in range(y_pred.shape[0]):
            if np.argmax(y_pred[i][:10]) == np.argmax(y_test[i]):
                acc = acc + 1

        return acc / y_pred.shape[0]


model = StudentModel()
model.train_model.summary()

lambda_const = 0

model.train_model.compile(
    optimizer=keras.optimizers.Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
    loss=lambda y_true, y_pred: knowledge_distillation_loss(y_true, y_pred, lambda_const),
    metrics=[accuracy, categorical_crossentropy, soft_logloss]
)

training_callback = TrainingCallback(model, 'distilled')


model.train_model.fit(
    x_train, y_train_,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test_),
    verbose=1, shuffle=True,
    callbacks=[
        training_callback
    ],
)



