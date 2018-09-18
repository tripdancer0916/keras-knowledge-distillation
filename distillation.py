# !/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import print_function
import keras
from keras import backend as K
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.engine.topology import Input, Container
from keras.engine.training import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, advanced_activations, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Convolution2D, pooling, Lambda
from keras.callbacks import ModelCheckpoint
from keras.models import load_model as keras_load_model
import os

temperature = 5.0
lambda_const = 0.1
batch_size = 100
num_classes = 10
epochs = 300
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


def distillation_loss(input):
    y_pred, y_true, teacher_probabilities_T, probabilities_T = input
    return lambda_const * K.categorical_crossentropy(y_true, y_pred) + \
           K.categorical_crossentropy(teacher_probabilities_T, probabilities_T)


teacher_model = keras_load_model(model_path)

input_layer = Input(x_train.shape[1:])
x = Convolution2D(32, (3, 3), padding='same')(input_layer)
x = BatchNormalization()(x)
x = advanced_activations.LeakyReLU(alpha=0.1)(x)
x = Convolution2D(32, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = advanced_activations.LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D((2, 2), strides=(2, 2))(x)
x = Dropout(0.5)(x)
x = Convolution2D(64, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = advanced_activations.LeakyReLU(alpha=0.1)(x)
x = Convolution2D(64, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = advanced_activations.LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D((2, 2), strides=(2, 2))(x)
x = Dropout(0.5)(x)
x = Convolution2D(64, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = advanced_activations.LeakyReLU(alpha=0.1)(x)

x = pooling.GlobalAveragePooling2D()(x)
logits = Dense(10, activation=None)(x)
output_softmax = Activation('softmax')(logits)

y_true = Input(name='y_true', shape=[None], dtype='float32')

output = Lambda(distillation_loss, output_shape=(1,))(
    [output_softmax, y_true, teacher_probabilities_T, probabilities_T]
)

opt = keras.optimizers.Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model = Model(input_layer, output)
model.summary()
model.compile(loss={'distillation_loss': lambda y_true, y_pred: y_pred},
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('Not using data augmentation.')

callbacks = [ModelCheckpoint(filepath="./models/model.ep{epoch:02d}.h5")]
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True, verbose=2, callbacks=callbacks)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
