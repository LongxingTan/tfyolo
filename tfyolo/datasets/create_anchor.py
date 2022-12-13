#! /usr/bin/env python
# coding=utf-8
# @Author: Longxing Tan, tanlongxing888@163.com
# Implementations of anchor generator that uses the k-means clustering to generate anchors for new data

import numpy as np


class Anchor(object):
    # create the default anchors by k-means
    def __init__(self):
        pass

    def kmeans(self, boxes, k, dist=np.mean):
        n_examples = boxes.shape[0]
        distances = np.empty((n_examples, k))
        last_clusters = np.zeros((n_examples,))

        clusters = boxes[np.random.choice(n_examples, k, replace=False)]
        while True:
            for example in range(n_examples):
                distances[example] = 1 - self.iou(boxes[example], clusters)

            nearest_clusters = np.argmin(distances, axis=1)
            if (last_clusters == nearest_clusters).all():
                break
            for cluster in range(k):
                clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)
            last_clusters = nearest_clusters

        return clusters

    def generate_anchor(self, annotations_dir, k=9):
        annotations = self.prepare_annotations(annotations_dir)
        clusters = self.kmeans(annotations, k=k)
        avg_iou = self.get_avg_iou(annotations, clusters)
        print("Average IOU", avg_iou)
        anchors = clusters.astype("int").tolist()
        anchors = sorted(anchors, key=lambda x: x[0] * x[1])
        return anchors

    def prepare_annotations(self, list_path):
        with open(list_path, "r") as f:
            annotations = [line.strip() for line in f.readlines() if len(line.strip().split()[1:]) != 0]
        print("load examples : {}".format(len(annotations)))

        result = []
        for i, annotation in enumerate(annotations):
            line = annotation.split()
            bboxes = np.array([list(map(float, box.split(","))) for box in line[1:]])
            assert bboxes.shape[1] == 5, "make sure the labeled objective has xmin,ymin,xmax,ymax,class"
            bbox_wh = bboxes[:, 2:4] - bboxes[:, 0:2]  # n_box * 2
            result.append(bbox_wh)
        result = np.concatenate(result, axis=0)
        return result

    def iou(self, box, clusters):
        """
        Calculates the Intersection over Union (IoU) between a box and k clusters.
        param:
            box: tuple or array, shifted to the origin (i. e. width and height)
            clusters: numpy array of shape (k, 2) where k is the number of clusters
        return:
            numpy array of shape (k, 0) where k is the number of clusters
        """
        x = np.minimum(clusters[:, 0], box[0])
        y = np.minimum(clusters[:, 1], box[1])
        if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
            raise ValueError("Box has no area")

        intersection = x * y
        box_area = box[0] * box[1]
        cluster_area = clusters[:, 0] * clusters[:, 1]

        iou_ = np.true_divide(intersection, box_area + cluster_area - intersection + 1e-7)
        # iou_ = intersection / (box_area + cluster_area - intersection + 1e-7)
        return iou_

    def get_avg_iou(self, boxes, clusters):
        return np.mean([np.max(self.iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


if __name__ == "__main__":
    anchor = Anchor()
    anchors = anchor.generate_anchor(annotations_dir="../data/voc2012/VOCdevkit/VOC2012/train.txt", k=9)
    print(anchors)
