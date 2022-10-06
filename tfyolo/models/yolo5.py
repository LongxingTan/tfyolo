#! /usr/bin/env python
# coding=utf-8
# @Author: Longxing Tan, tanlongxing888@163.com
# Implementations of Yolov5 main model

import math

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
import yaml

from tfyolo.layers.module import SPP, SPPCSP, Bottleneck, BottleneckCSP, BottleneckCSP2, Conv, DWConv, Focus, VoVCSP


class Yolo(object):
    def __init__(self, yaml_dir):
        with open(yaml_dir) as f:
            yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
        self.module_list = self.parse_model(yaml_dict)
        module = self.module_list[-1]
        if isinstance(module, Detect):
            # transfer the anchors to grid coordinator, 3 * 3 * 2
            module.anchors /= tf.reshape(module.stride, [-1, 1, 1])

    def __call__(self, img_size, name="yolo"):
        x = tf.keras.Input([img_size, img_size, 3])
        y = []
        for module in self.module_list:
            if module.f != -1:  # if not from previous layer
                if isinstance(module.f, int):
                    x = y[module.f]
                else:
                    x = [x if j == -1 else y[j] for j in module.f]

            x = module(x)
            y.append(x)
        return tf.keras.Model(inputs=x, outputs=y, name=name)

    def parse_model(self, yaml_dict):
        anchors, nc = yaml_dict["anchors"], yaml_dict["nc"]
        depth_multiple, width_multiple = yaml_dict["depth_multiple"], yaml_dict["width_multiple"]
        num_anchors = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
        output_dims = num_anchors * (nc + 5)

        layers = []
        # from, number, module, args
        for i, (f, number, module, args) in enumerate(yaml_dict["backbone"] + yaml_dict["head"]):
            # all component is a Class, initialize here, call in self.forward
            module = eval(module) if isinstance(module, str) else module

            for j, arg in enumerate(args):
                try:
                    args[j] = eval(arg) if isinstance(arg, str) else arg  # eval strings, like Detect(nc, anchors)
                except ValueError:
                    pass

            number = max(round(number * depth_multiple), 1) if number > 1 else number  # control the model scale

            if module in [Conv2D, Conv, Bottleneck, SPP, DWConv, Focus, BottleneckCSP, BottleneckCSP2, SPPCSP, VoVCSP]:
                c2 = args[0]
                c2 = math.ceil(c2 * width_multiple / 8) * 8 if c2 != output_dims else c2
                args = [c2, *args[1:]]

                if module in [BottleneckCSP, BottleneckCSP2, SPPCSP, VoVCSP]:
                    args.insert(1, number)
                    number = 1

            modules = tf.keras.Sequential(*[module(*args) for _ in range(number)]) if number > 1 else module(*args)
            modules.i, modules.f = i, f
            layers.append(modules)
        return layers


class Detect(tf.keras.layers.Layer):
    def __init__(self, num_classes, anchors=()):
        super(Detect, self).__init__()
        self.num_classes = num_classes
        self.num_scale = len(anchors)
        self.output_dims = self.num_classes + 5
        self.num_anchors = len(anchors[0]) // 2
        self.stride = np.array([8, 16, 32], np.float32)  # fixed here, modify if structure changes
        self.anchors = tf.cast(tf.reshape(anchors, [self.num_anchors, -1, 2]), tf.float32)
        self.modules = [Conv2D(self.output_dims * self.num_anchors, 1, use_bias=False) for _ in range(self.num_scale)]

    def call(self, x, training=True):
        res = []
        for i in range(self.num_scale):  # number of scale layer, default=3
            y = self.modules[i](x[i])
            _, grid1, grid2, _ = y.shape
            y = tf.reshape(y, (-1, grid1, grid2, self.num_scale, self.output_dims))

            grid_xy = tf.meshgrid(tf.range(grid1), tf.range(grid2))  # grid[x][y]==(y,x)
            grid_xy = tf.cast(tf.expand_dims(tf.stack(grid_xy, axis=-1), axis=2), tf.float32)

            y_norm = tf.sigmoid(y)  # sigmoid for all dims
            xy, wh, conf, classes = tf.split(y_norm, (2, 2, 1, self.num_classes), axis=-1)

            pred_xy = (xy * 2.0 - 0.5 + grid_xy) * self.stride[i]  # decode pred to xywh
            pred_wh = (wh * 2) ** 2 * self.anchors[i] * self.stride[i]

            out = tf.concat([pred_xy, pred_wh, conf, classes], axis=-1)
            res.append(out)
        return res
