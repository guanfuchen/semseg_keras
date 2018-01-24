#!/usr/bin/python
# -*- coding: UTF-8 -*-
from keras.metrics import binary_accuracy
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import os
import numpy as np
import argparse

# 数据集相关
from semseg_keras.dataloader.camvid_loader import camvidLoader

# 语义分割模型相关
from semseg_keras.modelloader.fcn import fcn32s
from semseg_keras.modelloader.densenet import DenseNetFCN
from semseg_keras.modelloader.unet import unet
from semseg_keras.modelloader.segnet import segnet

def train(args):
    img_height, img_width, img_channel = (224, 224, 3)

    batch_size = 3
    HOME_PATH = os.path.expanduser('~')
    local_path = os.path.join(HOME_PATH, 'Data/CamVid')
    dst = camvidLoader(local_path, is_transform=True, is_augment=True, batch_size=batch_size, img_size=(img_height, img_width))
    val_dst = camvidLoader(local_path, split='val', is_transform=True, is_augment=True, batch_size=batch_size, img_size=(img_height, img_width))

    input_shape = (img_height, img_width, img_channel)
    n_classes = dst.n_classes
    if args.structure == 'fcn32s':
        model = fcn32s(input_shape=input_shape, n_classes=n_classes)
    elif args.structure == 'unet':
        model = unet(input_shape=input_shape, n_classes=n_classes)
    elif args.structure == 'segnet':
        model = segnet(input_shape=input_shape, n_classes=n_classes)
    elif args.structure == 'DenseNetFCN':
        model = DenseNetFCN(input_shape, classes=n_classes, growth_rate=16, nb_layers_per_block=[4, 5, 7, 10, 12, 15], upsampling_type='deconv')
    optimizer = SGD(lr=0.001, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    callbacks = []
    model_checkpoint_filepath = "{}-weights-best.hdf5".format(args.structure)
    if os.path.exists(model_checkpoint_filepath):
        model.load_weights(model_checkpoint_filepath)
    model_checkpoint = ModelCheckpoint(filepath=model_checkpoint_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks.append(model_checkpoint)
    # 校准精度提升时存储模型ModelCheckPoint
    model.fit_generator(generator=dst.generate_batch_data(), validation_data=val_dst.generate_batch_data(), steps_per_epoch=dst.__len__()/batch_size, validation_steps=val_dst.__len__()/batch_size, epochs=1000, callbacks=callbacks, verbose=1)

if __name__=='__main__':
    # print('train----in----')
    parser = argparse.ArgumentParser(description='training parameter setting')
    parser.add_argument('--structure', type=str, default='fcn32s', help='use the net structure to segment [ fcn32s DenseNetFCN ]')
    parser.add_argument('--resume_model', type=str, default='', help='resume model path [ fcn32s_camvid_9.pkl ]')
    parser.add_argument('--resume_model_state_dict', type=str, default='', help='resume model state dict path [ fcn32s_camvid_9.pt ]')
    parser.add_argument('--save_model', type=bool, default=False, help='save model [ False ]')
    parser.add_argument('--save_epoch', type=int, default=1, help='save model after epoch [ 1 ]')
    parser.add_argument('--init_vgg16', type=bool, default=False, help='init model using vgg16 weights [ False ]')
    parser.add_argument('--dataset_path', type=str, default='', help='train dataset path [ /home/Data/CamVid ]')
    parser.add_argument('--data_augment', type=bool, default=False, help='enlarge the training data [ False ]')
    parser.add_argument('--batch_size', type=int, default=1, help='train dataset batch size [ 1 ]')
    parser.add_argument('--lr', type=float, default=1e-5, help='train learning rate [ 0.01 ]')
    parser.add_argument('--vis', type=bool, default=False, help='visualize the training results [ False ]')
    args = parser.parse_args()
    # print(args.resume_model)
    # print(args.save_model)
    print(args)
    train(args)
    # print('train----out----')