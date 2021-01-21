#! /usr/bin/env python
# coding=utf-8
# @Author: Longxing Tan, tanlongxing888@163.com

import sys
import os
filePath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.split(filePath)[0])

import tensorflow as tf
from configs.config import params
from dataset.load_data import DataLoader, TestDataLoader
from dataset.image_utils import box_iou, xyxy2xywh, resize_back
from model.metrics import ap_per_class
from model.post_process import batch_non_max_suppression


class Evaluator(object):
    def __init__(self, model_dir, num_classes, img_size, prediction_dir):
        self.model = tf.saved_model.load(model_dir)
        self.num_classes = num_classes
        self.img_size = img_size
        self.prediction_dir = prediction_dir

    def evaluate(self, test_dataset):
        results = []
        for step, image in enumerate(test_dataset):
            pred_bbox = self.model(image)
            pred_bbox = [tf.reshape(x, (tf.shape(x)[0], -1, tf.shape(x)[-1])) for x in pred_bbox]
            pred_bbox = tf.concat(pred_bbox, axis=1)  # batch_size * -1 * (num_class + 5)
            output = batch_non_max_suppression(pred_bbox)  # n_image list, [n_pred * 6, ...]

            for i, pred in enumerate(output):
                pred = resize_back(pred.numpy(), target_sizes=self.img_size)

