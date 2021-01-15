#! /usr/bin/env python
# coding=utf-8
# @Author: Longxing Tan, tanlongxing888@163.com

import numpy as np
import tensorflow as tf
from .label_anchor import AnchorLabeler


class DataLoader(object):
    def __init__(self, DataReader, anchors, stride, img_size=640, anchor_assign_method='wh',
                 anchor_positive_augment=True):
        '''
        data pipeline from data_reader (image,label) to tf.data
        '''
        self.data_reader = DataReader
        self.anchor_label = AnchorLabeler(anchors,
                                          grids=img_size / stride,
                                          img_size=img_size,
                                          assign_method=anchor_assign_method,
                                          extend_offset=anchor_positive_augment)

        self.img_size = img_size

    def __call__(self, batch_size=8, valid=False, test=False):
        if not test:
            dataset = tf.data.Dataset.from_generator(self.data_reader.iter,
                                                     output_types=(tf.float32, tf.float32),
                                                     output_shapes=([self.img_size, self.img_size, 3], [None, 5]))
        else:
            dataset = tf.data.Dataset.from_generator(self.data_reader.iter,
                                                     output_types=tf.float32,
                                                     output_shapes=[self.img_size, self.img_size, 3])

        if (not test) & (not valid):  # when train
            dataset = dataset.map(self.transform, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def transform(self, image, label):
        label_encoder = self.anchor_label.encode(label)
        return image, label_encoder

