"""
`YOLOv3: An Incremental Improvement
<https://arxiv.org/abs/1804.02767?e05802c1_page=1>`_
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D


class YoloV3(object):
    def __init__(self):
        self.backbone = Darknet53()

    def __call__(self, inputs):
        return


def conv_block():
    return


class Darknet53(tf.keras.layers.Layer):
    def __init__(self):
        pass

    def build(self):
        pass

    def call(self):
        return

    def get_config(self):
        pass
