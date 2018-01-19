#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import collections
import random

import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt
from PIL import Image

import keras.backend as K
from keras.utils.np_utils import to_categorical


class camvidLoader(object):
    def __init__(self, root, split="train", is_transform=False, is_augment=False, batch_size = 1, img_size=(360, 480)):
        super(camvidLoader, self).__init__()
        self.root = root
        self.split = split
        self.img_size = img_size
        self.is_transform = is_transform
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.n_classes = 13
        self.files = collections.defaultdict(list)
        self.joint_augment_transform = None
        self.is_augment = is_augment
        self.data_format = K.image_data_format()
        self.batch_size = batch_size
        assert self.data_format in {'channels_first', 'channels_last'}

        file_list = os.listdir(root + '/' + split)
        self.files[split] = file_list

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_name = self.files[self.split][index]
        img_path = self.root + '/' + self.split + '/' + img_name
        lbl_path = self.root + '/' + self.split + 'annot/' + img_name

        img = Image.open(img_path)
        lbl = Image.open(lbl_path)
        img = img.resize(self.img_size)
        lbl = lbl.resize(self.img_size)

        img = np.array(img, dtype=np.uint8)
        lbl = np.array(lbl, dtype=np.int32)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def get_all_data(self):
        all_data_len = self.__len__()
        imgs = []
        lbls = []
        for i in range(all_data_len):
            img, lbl = self.__getitem__(i)
            imgs.append(img)
            lbls.append(lbl)
        imgs = np.array(imgs)
        lbls = np.array(lbls)
        return imgs, lbls

    def generate_batch_data(self):

        while True:
            for i in range(0, self.__len__(), self.batch_size):
                imgs = []
                lbls = []
                for j in range(i, i+self.batch_size):
                    # print j
                    if j < self.__len__():
                        img, lbl = self.__getitem__(j)
                        imgs.append(img)
                        lbls.append(lbl)
                imgs = np.array(imgs)
                lbls = np.array(lbls)
                # print(imgs.shape, lbls.shape)
                yield imgs, lbls

    # 转换HWC为CHW
    def transform(self, img, lbl):
        img = img[:, :, ::-1]
        img = img.astype(np.float64)
        img -= self.mean
        img = img.astype(float) / 255.0
        # HWC -> CHW
        if self.data_format == 'channels_first':
            img = img.transpose(2, 0, 1)

        # print(to_categorical(lbl).reshape(480, 480, -1).shape)
        lbl = to_categorical(lbl, num_classes=13).reshape(self.img_size[0], self.img_size[1], -1)
        return img, lbl

    def decode_segmap(self, temp, plot=False):
        Sky = [128, 128, 128]
        Building = [128, 0, 0]
        Pole = [192, 192, 128]
        Road_marking = [255, 69, 0]
        Road = [128, 64, 128]
        Pavement = [60, 40, 222]
        Tree = [128, 128, 0]
        SignSymbol = [192, 128, 128]
        Fence = [64, 64, 128]
        Car = [64, 0, 128]
        Pedestrian = [64, 64, 0]
        Bicyclist = [0, 128, 192]
        Unlabelled = [0, 0, 0]

        label_colours = np.array([Sky, Building, Pole, Road_marking, Road,
                                  Pavement, Tree, SignSymbol, Fence, Car,
                                  Pedestrian, Bicyclist, Unlabelled])
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = label_colours[l, 0]
            g[temp == l] = label_colours[l, 1]
            b[temp == l] = label_colours[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

if __name__ == '__main__':
    HOME_PATH = os.path.expanduser('~')
    local_path = os.path.join(HOME_PATH, 'Data/CamVid')
    batch_size = 4
    dst = camvidLoader(local_path, is_transform=True, is_augment=True, batch_size=batch_size)
    dst.generate_batch_data()
