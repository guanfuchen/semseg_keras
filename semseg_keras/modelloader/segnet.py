#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, ZeroPadding2D, BatchNormalization, Activation
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras


def segnet(input_shape=None, n_classes=21):
    input = Input(shape=input_shape)

    # segnet中编码器结构
    zeropad1 = ZeroPadding2D((1, 1))(input)
    conv1 = Conv2D(64, 3, padding='valid', kernel_initializer='he_normal')(zeropad1)
    bn1 = BatchNormalization()(conv1)
    ac1 = Activation('relu')(bn1)
    pool1 = MaxPooling2D((2, 2))(ac1)

    zeropad2 = ZeroPadding2D((1, 1))(pool1)
    conv2 = Conv2D(128, 3, padding='valid', kernel_initializer='he_normal')(zeropad2)
    bn2 = BatchNormalization()(conv2)
    ac2 = Activation('relu')(bn2)
    pool2 = MaxPooling2D((2, 2))(ac2)

    zeropad3 = ZeroPadding2D((1, 1))(pool2)
    conv3 = Conv2D(256, 3, padding='valid', kernel_initializer='he_normal')(zeropad3)
    bn3 = BatchNormalization()(conv3)
    ac3 = Activation('relu')(bn3)
    pool3 = MaxPooling2D((2, 2))(ac3)

    zeropad4 = ZeroPadding2D((1, 1))(pool3)
    conv4 = Conv2D(512, 3, padding='valid', kernel_initializer='he_normal')(zeropad4)
    bn4 = BatchNormalization()(conv4)
    ac4 = Activation('relu')(bn4)

    # segnet中解码器结构
    zeropad5 = ZeroPadding2D((1, 1))(ac4)
    conv5 = Conv2D(512, 3, padding='valid', kernel_initializer='he_normal')(zeropad5)
    bn5 = BatchNormalization()(conv5)

    unpool6 = UpSampling2D(size=(2, 2))(bn5)
    zeropad6 = ZeroPadding2D((1, 1))(unpool6)
    conv6 = Conv2D(256, 3, padding='valid', kernel_initializer='he_normal')(zeropad6)
    bn6 = BatchNormalization()(conv6)

    unpool7 = UpSampling2D(size=(2, 2))(bn6)
    zeropad7 = ZeroPadding2D((1, 1))(unpool7)
    conv7 = Conv2D(128, 3, padding='valid', kernel_initializer='he_normal')(zeropad7)
    bn7 = BatchNormalization()(conv7)

    unpool8 = UpSampling2D(size=(2, 2))(bn7)
    zeropad8 = ZeroPadding2D((1, 1))(unpool8)
    conv8 = Conv2D(64, 3, padding='valid', kernel_initializer='he_normal')(zeropad8)
    bn8 = BatchNormalization()(conv8)

    conv9 = Conv2D(n_classes, 1, padding='valid', kernel_initializer='he_normal')(bn8)
    ac9 = Activation('softmax')(conv9)

    model = Model(input=input, output=ac9)

    return model

if __name__ == '__main__':
    input_shape = (360, 480, 3)
    n_classes = 13
    model = segnet(input_shape=input_shape, n_classes=n_classes)
    model.summary()
