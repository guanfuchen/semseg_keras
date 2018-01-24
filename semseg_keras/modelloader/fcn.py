#!/usr/bin/python
# -*- coding: UTF-8 -*-

from keras.layers import Conv2D, MaxPooling2D, Dropout, Input, Layer, InputSpec, Activation
from keras.layers.convolutional import UpSampling2D

from keras.regularizers import l2
from keras.models import Model
import keras.backend as K
import numpy as np


def fcn32s(input_shape=None, n_classes=21):
    # 输入数据
    input = Input(shape=input_shape)

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Convolutional layers transfered from fully-connected layers
    x = Conv2D(4096, (7, 7), activation='relu', padding='same', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2')(x)
    x = Dropout(0.5)(x)
    #classifying layer
    x = Conv2D(n_classes, (1, 1), kernel_initializer='he_normal', activation='linear', padding='valid', strides=(1, 1))(x)

    x = UpSampling2D(size=(32, 32))(x)

    scores = Activation('softmax')(x)
    # print(scores)

    model = Model(inputs=input, outputs=scores)
    return model

if __name__ == '__main__':
    input_shape = (224, 224, 3)
    n_classes = 13
    model = fcn32s(input_shape=input_shape, n_classes=n_classes)