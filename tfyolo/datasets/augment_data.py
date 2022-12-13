#! /usr/bin/env python
# coding=utf-8
# @Author: Longxing Tan, tanlongxing888@163.com

import math
import random

import cv2
import numpy as np

from .image_utils import resize_image

random.seed(1919)


def load_mosaic_image(index, mosaic_border, image_target_size, images_dir, labels):
    # labels style: pixel or norm
    # labels output: pixel
    max_index = len(labels) - 1
    indices = [index] + [random.randint(0, max_index) for _ in range(3)]
    yc, xc = [int(random.uniform(-i, 2 * image_target_size + i)) for i in mosaic_border]  # mosaic center x, y
    label_mosaic = []

    for i, index in enumerate(indices):
        img_dir = images_dir[index]
        img = cv2.imread(img_dir)
        label = labels[index].copy()
        h_origin, w_origin, _ = img.shape

        img = resize_image(img, target_sizes=image_target_size, keep_ratio=False)
        h, w, _ = img.shape

        if i == 0:  # top left
            img_mosaic = np.full(
                (image_target_size * 2, image_target_size * 2, 3), 128, dtype=np.uint8
            )  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, image_target_size * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(image_target_size * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, image_target_size * 2), min(image_target_size * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img_mosaic[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
        padw = x1a - x1b
        padh = y1a - y1b

        label_new = label.copy()
        if label.size > 0:
            if np.max(label_new[:, 0:4]) > 1:  # if label is pixel, [0, size]
                label_new[:, [0, 2]] = label_new[:, [0, 2]] / w_origin * w + padw
                label_new[:, [1, 3]] = label_new[:, [1, 3]] / h_origin * h + padh
            else:  # if label is normed, [0, 1]
                label_new[:, [0, 2]] = label_new[:, [0, 2]] * w + padw
                label_new[:, [1, 3]] = label_new[:, [1, 3]] * h + padh
        label_mosaic.append(label_new)

    if len(label_mosaic):
        label_mosaic = np.concatenate(label_mosaic, 0)
        label_mosaic[:, :4] = np.clip(label_mosaic[:, :4], 0, 2 * image_target_size)

    img_mosaic, label_mosaic = random_perspective(img_mosaic, label=label_mosaic, border=mosaic_border)
    return img_mosaic, label_mosaic


def random_perspective(img, label=(), degrees=10, translate=0.1, scale=0.1, shear=10, perspective=0.0, border=(0, 0)):
    # labels style: pixel, [xyxy, cls]
    img = img.astype(np.uint8)

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(label)
    if n:
        if np.max(label[:, 0:4]) <= 1.0:  # transfer to pixel level
            label[:, [0, 2]] = label[:, [0, 2]] * img.shape[1]
            label[:, [1, 3]] = label[:, [1, 3]] * img.shape[0]
        # assert np.max(labels[:, 0:4]) > 1, "don't use norm box coordinates here"
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = label[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1

        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # reject warped points outside of image
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        area0 = (label[:, 2] - label[:, 0]) * (label[:, 3] - label[:, 1])
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
        i = (w > 2) & (h > 2) & (area / (area0 * scale + 1e-16) > 0.2) & (ar < 20)

        label = label[i]
        label[:, 0:4] = xy[i]

        if label.size == 0:  # in case, all labels is out
            label = np.array([[0, 0, 0, 0, 0]], np.float32)
    return img, label


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    rand = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * rand[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * rand[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * rand[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


def random_flip(img, labels=None):
    # Please note the labels should be normalized into [0, 1]
    # assert np.max(labels) <= 1, "The flip labels should be normalized [0, 1]"
    if np.max(labels[:, 0:4]) > 1:  # transfer to pixel level
        labels[:, [0, 2]] = labels[:, [0, 2]] / img.shape[1]
        labels[:, [1, 3]] = labels[:, [1, 3]] / img.shape[0]

    lr_flip = True
    if lr_flip and random.random() < 0.5:
        img = np.fliplr(img)
        if labels is not None:
            labels[:, [0, 2]] = 1 - labels[:, [0, 2]]

    ud_flip = False
    if ud_flip and random.random() < 0.5:
        img = np.flipud(img)
        if labels is not None:
            labels[:, [1, 3]] = 1 - labels[:, [1, 3]]
    return img, labels
