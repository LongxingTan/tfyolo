#! /usr/bin/env python
# coding=utf-8
# @Author: Longxing Tan, tanlongxing888@163.com

import cv2
import numpy as np
import tensorflow as tf


def resize_image(img, target_sizes, keep_ratio=True, label=None):
    # Please Note： label style should be normalized xyxy, otherwise need modify
    # if keep_ratio is True, letterbox using padding
    if not isinstance(target_sizes, (list, set, tuple)):
        target_sizes = [target_sizes, target_sizes]
    target_h, target_w = target_sizes

    h, w, _ = img.shape
    scale = min(target_h / h, target_w / w)
    temp_h, temp_w = int(scale * h), int(scale * w)
    image_resize = cv2.resize(img, (temp_w, temp_h), interpolation=cv2.INTER_CUBIC)

    if keep_ratio:
        image_new = np.full(shape=(target_h, target_w, 3), fill_value=128.0, dtype="uint8")
        delta_h, delta_w = (target_h - temp_h) // 2, (target_w - temp_w) // 2
        image_new[delta_h : delta_h + temp_h, delta_w : delta_w + temp_w, :] = image_resize.copy()

        if label is not None:
            label[:, [0, 2]] = (label[:, [0, 2]] * scale * w + delta_w) / target_w
            label[:, [1, 3]] = (label[:, [1, 3]] * scale * h + delta_h) / target_h
            return image_new, label
        else:
            return image_new
    else:
        if label is not None:
            # it's fine if the label is normalized and the image is cv2.resize directly
            return image_resize, label
        else:
            return image_resize


def resize_back(bboxes, target_sizes, original_shape):
    original_h, original_w = original_shape[:2]

    resize_ratio = min(target_sizes / original_w, target_sizes / original_h)
    dw = (target_sizes - resize_ratio * original_w) / 2
    dh = (target_sizes - resize_ratio * original_h) / 2
    bboxes[:, [0, 2]] = 1.0 * (bboxes[:, [0, 2]] - dw) / resize_ratio
    bboxes[:, [1, 3]] = 1.0 * (bboxes[:, [1, 3]] - dh) / resize_ratio
    return bboxes


def xyxy2xywh(box):
    y0 = (box[:, 0:1] + box[:, 2:3]) / 2.0  # x center
    y1 = (box[:, 1:2] + box[:, 3:4]) / 2.0  # y center
    y2 = box[:, 2:3] - box[:, 0:1]  # width
    y3 = box[:, 3:4] - box[:, 1:2]  # height
    y = (
        tf.concat([y0, y1, y2, y3], axis=-1)
        if isinstance(box, tf.Tensor)
        else np.concatenate([y0, y1, y2, y3], axis=-1)
    )
    return y


def xywh2xyxy(box):
    y0 = box[..., 0:1] - box[..., 2:3] / 2  # top left x
    y1 = box[..., 1:2] - box[..., 3:4] / 2  # top left y
    y2 = box[..., 0:1] + box[..., 2:3] / 2  # bottom right x
    y3 = box[..., 1:2] + box[..., 3:4] / 2  # bottom right y
    y = (
        tf.concat([y0, y1, y2, y3], axis=-1)
        if isinstance(box, tf.Tensor)
        else np.concatenate([y0, y1, y2, y3], axis=-1)
    )
    return y


def box_iou(box1, box2, broadcast=True):
    # input: xywh, n * 4, m * 4
    # output: n * m
    if broadcast:
        box1 = tf.expand_dims(box1, 1)  # n * 1 * 4
        box2 = tf.expand_dims(box2, 0)  # 1 * m * 4
    boxes1_area = box1[..., 2] * box1[..., 3]
    boxes2_area = box2[..., 2] * box2[..., 3]

    box1 = tf.concat(
        [box1[..., :2] - box1[..., 2:] * 0.5, box1[..., :2] + box1[..., 2:] * 0.5], axis=-1
    )  # xmin, ymin, xmax, ymax
    box2 = tf.concat([box2[..., :2] - box2[..., 2:] * 0.5, box2[..., :2] + box2[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(box1[..., :2], box2[..., :2])
    right_down = tf.minimum(box1[..., 2:], box2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 1e-6)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area + 1e-9
    iou = 1.0 * inter_area / union_area
    return iou
