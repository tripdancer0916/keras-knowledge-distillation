# !/usr/bin/env python
# -*- coding:utf-8 -*-

from keras.callbacks import Callback

class TrainingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
