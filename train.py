#!/usr/bin/python
# -*- coding: UTF-8 -*-
from keras.metrics import binary_accuracy
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import os
import numpy as np

from semseg_keras.modelloader.fcn import FCN_Vgg16_32s
from semseg_keras.dataloader.camvid_loader import camvidLoader
from semseg_keras.loss import binary_crossentropy_with_logits

if __name__ == '__main__':
    img_height, img_width = (224, 224)
    batch_size = 3
    HOME_PATH = os.path.expanduser('~')
    local_path = os.path.join(HOME_PATH, 'cgf/Data/CamVid')
    dst = camvidLoader(local_path, is_transform=True, is_augment=True, batch_size=batch_size, img_size=(img_height, img_width))
    val_dst = camvidLoader(local_path, split='val', is_transform=True, is_augment=True, batch_size=batch_size, img_size=(img_height, img_width))

    input_shape = (img_height, img_width, 3)
    n_classes = dst.n_classes
    model = FCN_Vgg16_32s(input_shape=input_shape, n_classes=dst.n_classes)
    optimizer = SGD(lr=0.001, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    callbacks = []
    # model_checkpoint_filepath = "FCN_Vgg16_32s-weights-{epoch:02d}-{val_acc:.2f}.hdf5"
    model_checkpoint_filepath = "FCN_Vgg16_32s-weights-best.hdf5"
    if os.path.exists(model_checkpoint_filepath):
        model.load_weights(model_checkpoint_filepath)
    model_checkpoint = ModelCheckpoint(filepath=model_checkpoint_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks.append(model_checkpoint)
    # 校准精度提升时存储模型ModelCheckPoint
    model.fit_generator(generator=dst.generate_batch_data(), validation_data=val_dst.generate_batch_data(), steps_per_epoch=dst.__len__()/batch_size, validation_steps=val_dst.__len__()/batch_size, epochs=1000, callbacks=callbacks, verbose=1)
