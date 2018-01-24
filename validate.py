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
from semseg_keras.metrics import scores

if __name__ == '__main__':
    img_height, img_width = (224, 224)
    batch_size = 3
    HOME_PATH = os.path.expanduser('~')
    local_path = os.path.join(HOME_PATH, 'cgf/Data/CamVid')
    dst = camvidLoader(local_path, split='test', is_transform=True, is_augment=True, batch_size=batch_size, img_size=(img_height, img_width))

    input_shape = (img_height, img_width, 3)
    n_classes = dst.n_classes
    model = FCN_Vgg16_32s(input_shape=input_shape, n_classes=dst.n_classes)
    optimizer = SGD(lr=0.001, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model_checkpoint_filepath = "FCN_Vgg16_32s-weights-best.hdf5"
    if os.path.exists(model_checkpoint_filepath):
        model.load_weights(model_checkpoint_filepath)
    # 校准精度提升时存储模型ModelCheckPoint
    # model.fit_generator(generator=dst.generate_batch_data(), validation_data=val_dst.generate_batch_data(), steps_per_epoch=dst.__len__()/batch_size, validation_steps=val_dst.__len__()/batch_size, epochs=1000, callbacks=callbacks, verbose=1)
    gts, preds = [], []
    imgs, labels = dst.get_all_data()
    pred_labels = model.predict(imgs)

    labels_argmax = np.argmax(labels, axis=3)
    pred_labels_argmax = np.argmax(pred_labels, axis=3)
    # print(labels.shape)
    # print(pred_labels.shape)
    # print(labels_argmax.shape)
    # print(pred_labels_argmax.shape)

    for dst_num in range(dst.__len__()):
    # for dst_num in range(3):
        label = np.array(labels_argmax[dst_num], np.int32)
        pred_label = np.array(pred_labels_argmax[dst_num], np.int32)
        label = np.expand_dims(label, axis=0)
        pred_label = np.expand_dims(pred_label, axis=0)
        # print(label.shape)
        # print(pred_label.shape)
        gts.append(label)
        preds.append(pred_label)

    # 输入scores中的数据是[label1, label2, ...] [pred1, pred2, ...]，而且label.shape=(1, height, weight)，pred.shape=(1, height, weight)
    score, class_iou = scores(gts, preds, n_class=dst.n_classes)
    for k, v in score.items():
        print(k, v)

    for i in range(dst.n_classes):
        print(i, class_iou[i])
