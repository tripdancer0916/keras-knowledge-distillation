# !/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
import keras
import argparse
import h5py
from keras import optimizers
from keras import backend as K
from keras.losses import categorical_crossentropy as logloss
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from keras.datasets import cifar10, cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import Iterator
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
from keras.utils import multi_gpu_model
from tensorflow.python.client import device_lib

batch_size = 128
num_classes = 100
epochs = 300

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


def knowledge_distillation_loss(input_distillation):
    y_pred, y_true, y_soft, y_pred_soft = input_distillation
    return (1 - args.lambda_const) * logloss(y_true, y_pred) + \
           args.lambda_const * args.temperature * args.temperature * logloss(y_soft, y_pred_soft)


class MyIterator(object):
    def __init__(self, iterator_org):
        self.iterator = iterator_org

    def __iter__(self):
        return self

    def __next__(self):
        tmp = next(self.iterator)
        return [tmp[0], tmp[1]], tmp[1]


class TrainingCallback(Callback):
    def __init__(self, model, model_prefix):
        super(TrainingCallback, self).__init__()
        self.model = model
        self.model_prefix = model_prefix

    def on_epoch_end(self, epoch, logs=None):
        acc = model.evaluate()
        print('val_acc: ', acc)
        time_stamp = datetime.strftime(datetime.now(pytz.timezone('Japan')), '%m%d%H%M')
        model_name = '{}_{}_epoch_{:03d}_ACC_{:.4f}_loss_{:.4f}.aiinside'.format(
            time_stamp, self.model_prefix, epoch + 1, acc, logs['loss'])
        save_model_path = os.path.join('./born_again_models', model_name)

        model.born_again_model.save(save_model_path)


class BornAgainModel(object):
    def __init__(self, teacher_model):
        self.train_model, self.born_again_model = None, None
        self.temperature = args.temperature
        self.teacher_model = keras_load_model(teacher_model)
        for i in range(len(self.teacher_model.layers)):
            self.teacher_model.layers[i].trainable = False
        self.teacher_model.compile(optimizer="adam", loss="categorical_crossentropy")
        self.train_model, self.born_again_model = self.prepare()
        self.train_model = convert_gpu_model(self.train_model)

    def prepare(self):
        self.teacher_model.layers.pop()
        input_layer = self.teacher_model.input
        teacher_logits = self.teacher_model.layers[-1].output
        teacher_logits_T = Lambda(lambda x: x / self.temperature)(teacher_logits)
        teacher_probabilities_T = Activation('softmax', name='softmax1_')(teacher_logits_T)

        x = Convolution2D(64, (3, 3), padding='same', name='conv2d1')(input_layer)
        x = BatchNormalization(name='bn1')(x)
        x = advanced_activations.LeakyReLU(alpha=0.1, name='lrelu1')(x)
        x = Convolution2D(64, (3, 3), padding='same', name='conv2d2')(x)
        x = BatchNormalization(name='bn2')(x)
        x = advanced_activations.LeakyReLU(alpha=0.1, name='lrelu2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)
        x = Dropout(0.3, name='drop1')(x)
        x = Convolution2D(128, (3, 3), padding='same', name='conv2d3')(x)
        x = BatchNormalization(name='bn3')(x)
        x = advanced_activations.LeakyReLU(alpha=0.1, name='lrelu3')(x)
        x = Convolution2D(128, (3, 3), padding='same', name='conv2d4')(x)
        x = BatchNormalization(name='bn4')(x)
        x = advanced_activations.LeakyReLU(alpha=0.1, name='lrelu4')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)
        x = Dropout(0.3, name='drop2')(x)
        x = Convolution2D(256, (3, 3), padding='same', name='conv2d5')(x)
        x = BatchNormalization(name='bn5')(x)
        x = advanced_activations.LeakyReLU(alpha=0.1, name='lrelu5')(x)
        x = Convolution2D(256, (3, 3), padding='same', name='conv2d6')(x)
        x = BatchNormalization(name='bn6')(x)
        x = advanced_activations.LeakyReLU(alpha=0.1, name='lrelu6')(x)
        x = Convolution2D(256, (3, 3), padding='same', name='conv2d7')(x)
        x = BatchNormalization(name='bn7')(x)
        x = advanced_activations.LeakyReLU(alpha=0.1, name='lrelu7')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)
        x = Dropout(0.3, name='drop3')(x)
        x = Flatten(name='flatten1')(x)
        x = Dense(512, activation=None, name='dense1')(x)
        x = BatchNormalization(name='bn8')(x)
        x = advanced_activations.LeakyReLU(alpha=0.1, name='lrelu8')(x)

        logits = Dense(num_classes, activation=None, name='dense2')(x)
        output_softmax = Activation('softmax', name='output_softmax')(logits)
        logits_T = Lambda(lambda x: x / self.temperature, name='logits')(logits)
        probabilities_T = Activation('softmax', name='probabilities')(logits_T)

        with tf.device('/cpu:0'):
            born_again_model = Model(inputs=input_layer, outputs=output_softmax)
            input_true = Input(name='input_true', shape=[None], dtype='float32')
        output_loss = Lambda(knowledge_distillation_loss, output_shape=(1,), name='kd_')(
            [output_softmax, input_true, teacher_probabilities_T, probabilities_T]
        )
        inputs = [input_layer, input_true]

        with tf.device('/cpu:0'):
            train_model = Model(inputs=inputs, outputs=output_loss)

        return train_model, born_again_model

    def evaluate(self):
        y_pred = self.born_again_model.predict(x_test)
        acc = 0
        for i in range(y_pred.shape[0]):
            if np.argmax(y_pred[i][:num_classes]) == np.argmax(y_test[i]):
                acc = acc + 1

        return acc / y_pred.shape[0]


def convert_gpu_model(org_model: Model) -> Model:
    gpu_count = len(device_lib.list_local_devices()) - 1
    if gpu_count > 1:
        train_model = multi_gpu_model(org_model, gpu_count)
    else:
        train_model = org_model
    return train_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Born Again Neural Networks for CIFAR-100')
    parser.add_argument('--temperature', type=float, default=10.0)
    parser.add_argument('--lambda_const', type=float, default=0.9)
    parser.add_argument('--teacher_model_path', type=str, default=None)

    args = parser.parse_args()

    model = BornAgainModel(args.teacher_model_path)
    model.born_again_model.summary()

    model.train_model.compile(
        optimizer=keras.optimizers.Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
        loss=lambda y_true, y_pred: y_pred,
    )

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.1,  # set range for random shear
        zoom_range=0.2,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None)

    datagen.fit(x_train)

    training_callback = TrainingCallback(model, 'Born-Again')

    tmp_iterator = datagen.flow(x_train, y_train, batch_size=batch_size)
    iterator = MyIterator(tmp_iterator)
    model.train_model.fit_generator(iterator,
                                    steps_per_epoch=x_train.shape[0] // batch_size,
                                    epochs=epochs,
                                    workers=4, callbacks=[training_callback])
