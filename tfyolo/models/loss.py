#! /usr/bin/env python
# coding=utf-8
# @Author: Longxing Tan, tanlongxing888@163.com
# Implementations of yolo loss function

import math

import tensorflow as tf


class YoloLoss(object):
    def __init__(self, anchors, ignore_iou_threshold, num_classes, img_size, label_smoothing=0):
        self.anchors = anchors
        self.strides = [8, 16, 32]
        self.ignore_iou_threshold = ignore_iou_threshold
        self.num_classes = num_classes
        self.img_size = img_size
        self.bce_conf = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.bce_class = tf.keras.losses.BinaryCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE, label_smoothing=label_smoothing
        )

    def __call__(self, y_true, y_pred):
        iou_loss_all = obj_loss_all = class_loss_all = 0
        balance = [1.0, 1.0, 1.0] if len(y_pred) == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6

        for i, (pred, true) in enumerate(zip(y_pred, y_true)):
            # preprocess, true: batch_size * grid * grid * 3 * 6, pred: batch_size * grid * grid * clss+5
            true_box, true_obj, true_class = tf.split(true, (4, 1, -1), axis=-1)
            pred_box, pred_obj, pred_class = tf.split(pred, (4, 1, -1), axis=-1)
            if tf.shape(true_class)[-1] == 1 and self.num_classes > 1:
                true_class = tf.squeeze(
                    tf.one_hot(tf.cast(true_class, tf.dtypes.int32), depth=self.num_classes, axis=-1), -2
                )

            # prepare: higher weights to smaller box, true_wh should be normalized to (0,1)
            box_scale = 2 - 1.0 * true_box[..., 2] * true_box[..., 3] / (self.img_size**2)
            obj_mask = tf.squeeze(true_obj, -1)  # obj or noobj, batch_size * grid * grid * anchors_per_grid
            background_mask = 1.0 - obj_mask
            conf_focal = tf.squeeze(tf.math.pow(true_obj - pred_obj, 2), -1)

            # iou/ giou/ ciou/ diou loss
            iou = bbox_iou(pred_box, true_box, xyxy=False, giou=True)
            iou_loss = (1 - iou) * obj_mask * box_scale  # batch_size * grid * grid * 3

            # confidence loss, Todo: multiply the iou
            conf_loss = self.bce_conf(true_obj, pred_obj)
            conf_loss = conf_focal * (obj_mask * conf_loss + background_mask * conf_loss)  # batch * grid * grid * 3

            # class loss
            # use binary cross entropy loss for multi class, so every value is independent and sigmoid
            # please note that the output of tf.keras.losses.bce is original dim minus the last one
            class_loss = obj_mask * self.bce_class(true_class, pred_class)

            iou_loss = tf.reduce_mean(tf.reduce_sum(iou_loss, axis=[1, 2, 3]))
            conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3]))
            class_loss = tf.reduce_mean(tf.reduce_sum(class_loss, axis=[1, 2, 3]))

            iou_loss_all += iou_loss * balance[i]
            obj_loss_all += conf_loss * balance[i]
            class_loss_all += class_loss * self.num_classes * balance[i]  # to balance the 3 loss

        try:
            print(
                "-" * 55,
                "iou",
                tf.reduce_sum(iou_loss_all).numpy(),
                ", conf",
                tf.reduce_sum(obj_loss_all).numpy(),
                ", class",
                tf.reduce_sum(class_loss_all).numpy(),
            )
        except ValueError:  # tf graph mode
            pass
        return (iou_loss_all, obj_loss_all, class_loss_all)


def bbox_iou(bbox1, bbox2, xyxy=False, giou=False, diou=False, ciou=False, epsilon=1e-9):
    assert bbox1.shape == bbox2.shape
    # giou loss: https://arxiv.org/abs/1902.09630
    if xyxy:
        b1x1, b1y1, b1x2, b1y2 = bbox1[..., 0], bbox1[..., 1], bbox1[..., 2], bbox1[..., 3]
        b2x1, b2y1, b2x2, b2y2 = bbox2[..., 0], bbox2[..., 1], bbox2[..., 2], bbox2[..., 3]
    else:  # xywh -> xyxy
        b1x1, b1x2 = bbox1[..., 0] - bbox1[..., 2] / 2, bbox1[..., 0] + bbox1[..., 2] / 2
        b1y1, b1y2 = bbox1[..., 1] - bbox1[..., 3] / 2, bbox1[..., 1] + bbox1[..., 3] / 2
        b2x1, b2x2 = bbox2[..., 0] - bbox2[..., 2] / 2, bbox2[..., 0] + bbox2[..., 2] / 2
        b2y1, b2y2 = bbox2[..., 1] - bbox2[..., 3] / 2, bbox2[..., 1] + bbox2[..., 3] / 2

    # intersection area
    inter = tf.maximum(tf.minimum(b1x2, b2x2) - tf.maximum(b1x1, b2x1), 0) * tf.maximum(
        tf.minimum(b1y2, b2y2) - tf.maximum(b1y1, b2y1), 0
    )

    # union area
    w1, h1 = b1x2 - b1x1 + epsilon, b1y2 - b1y1 + epsilon
    w2, h2 = b2x2 - b2x1 + epsilon, b2y2 - b2y1 + epsilon
    union = w1 * h1 + w2 * h2 - inter + epsilon

    # iou
    iou = inter / union

    if giou or diou or ciou:
        # enclosing box
        cw = tf.maximum(b1x2, b2x2) - tf.minimum(b1x1, b2x1)
        ch = tf.maximum(b1y2, b2y2) - tf.minimum(b1y1, b2y1)
        if giou:
            enclose_area = cw * ch + epsilon
            giou = iou - 1.0 * (enclose_area - union) / enclose_area
            return tf.clip_by_value(giou, -1, 1)
        if diou or ciou:
            c2 = cw**2 + ch**2 + epsilon
            rho2 = ((b2x1 + b2x2) - (b1x1 + b1x2)) ** 2 / 4 + ((b2y1 + b2y2) - (b1y1 + b1y2)) ** 2 / 4
            if diou:
                return iou - rho2 / c2
            elif ciou:
                v = (4 / math.pi**2) * tf.pow(tf.atan(w2 / h2) - tf.atan(w1 / h1), 2)
                alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)
    return tf.clip_by_value(iou, 0, 1)
