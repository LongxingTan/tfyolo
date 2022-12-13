#! /usr/bin/env python
# coding=utf-8
# @Author: Longxing Tan, tanlongxing888@163.com
# Implementations of anchor labeler that can assign the label to corresponding grids and encode the label

import tensorflow as tf


class AnchorLabeler(object):
    # transfer the annotated label to model target by anchor encoding, to calculate anchor based loss next step
    def __init__(
        self,
        anchors,
        grids,
        img_size=640,
        assign_method="wh",
        extend_offset=True,
        rect_style="rect4",
        anchor_match_threshold=4.0,
    ):  # 4.0 or 0.3
        self.anchors = anchors  # from yaml.anchors to Detect.anchors, w/h based on grid coordinators
        self.grids = grids
        self.img_size = img_size
        self.assign_method = assign_method
        self.extend_offset = extend_offset
        self.rect_style = rect_style
        self.anchor_match_threshold = anchor_match_threshold

    def encode(self, labels):
        """This is important for Yolo series.
        key part is: assign the label to which anchor and which grid,
        new encoding method of V4 solved the grid sensitivity problem
        labels: (n_bs * n_gt * 5), x/y/w/h/class, normalized image coordinators
        anchors: (3 * 3 * 2), scale * anchor_per_scale * wh,
        return: [[], [], []]
        """
        self.num_scales = self.anchors.shape[0]
        self.n_anchor_per_scale = self.anchors.shape[1]
        y_anchor_encode = []
        gain = tf.ones(5, tf.float32)

        for i in range(self.num_scales):
            anchor = self.anchors[i]
            grid_size = tf.cast(self.grids[i], tf.int32)
            y_true = tf.zeros([grid_size, grid_size, self.n_anchor_per_scale, 6], tf.float32)
            gain = tf.tensor_scatter_nd_update(gain, [[0], [1], [2], [3]], [grid_size] * 4)
            scaled_labels = labels * gain  # label coordinator now is the same with anchors

            if labels is not None:
                gt_wh = scaled_labels[..., 2:4]  # n_gt * 2
                if self.assign_method == "wh":
                    assert self.anchor_match_threshold > 1, "threshold is totally different for wh and iou assign"
                    matched_matrix = self.assign_criterion_wh(gt_wh, anchor, self.anchor_match_threshold)
                elif self.assign_method == "iou":
                    assert self.anchor_match_threshold < 1, "threshold is totally different for wh and iou assign"
                    matched_matrix = self.assign_criterion_iou(gt_wh, anchor, self.anchor_match_threshold)
                else:
                    raise ValueError

                n_gt = tf.shape(gt_wh)[0]
                assigned_anchor = tf.tile(
                    tf.reshape(tf.range(self.n_anchor_per_scale), (self.n_anchor_per_scale, 1)), (1, n_gt)
                )

                assigned_anchor = tf.expand_dims(assigned_anchor[matched_matrix], 1)  # filter
                assigned_anchor = tf.cast(assigned_anchor, tf.int32)

                assigned_label = tf.tile(tf.expand_dims(scaled_labels, 0), [self.n_anchor_per_scale, 1, 1])
                assigned_label = assigned_label[matched_matrix]

                if self.extend_offset:
                    assigned_label, assigned_anchor, grid_offset = self.enrich_pos_by_position(
                        assigned_label, assigned_anchor, gain, matched_matrix
                    )
                else:
                    grid_offset = tf.zeros_like(assigned_label[:, 0:2])

                assigned_grid = tf.cast(assigned_label[..., 0:2] - grid_offset, tf.int32)  # n_matched * 2
                assigned_grid = tf.clip_by_value(assigned_grid, clip_value_min=0, clip_value_max=grid_size - 1)

                # tensor: grid * grid * 3 * 6, indices（sparse index）: ~n_gt * gr * gr * 3, updates: ~n_gt * 6
                assigned_indices = tf.concat([assigned_grid[:, 1:2], assigned_grid[:, 0:1], assigned_anchor], axis=1)

                xy, wh, clss = tf.split(assigned_label, (2, 2, 1), axis=-1)
                xy = xy / gain[0] * self.img_size
                wh = wh / gain[1] * self.img_size
                obj = tf.ones_like(clss)
                assigned_updates = tf.concat([xy, wh, obj, clss], axis=-1)

                y_true = tf.tensor_scatter_nd_update(y_true, assigned_indices, assigned_updates)
            y_anchor_encode.append(y_true)
        return tuple(y_anchor_encode)  # add a tuple is important here, otherwise raise an error

    def assign_criterion_wh(self, gt_wh, anchors, anchor_threshold):
        # return: please note that the v5 default anchor_threshold is 4.0, related to the positive sample augment
        gt_wh = tf.expand_dims(gt_wh, 0)  # => 1 * n_gt * 2
        anchors = tf.expand_dims(anchors, 1)  # => n_anchor * 1 * 2
        ratio = gt_wh / anchors  # => n_anchor * n_gt * 2
        matched_matrix = (
            tf.reduce_max(tf.math.maximum(ratio, 1 / ratio), axis=2) < anchor_threshold
        )  # => n_anchor * n_gt
        return matched_matrix

    def assign_criterion_iou(self, gt_wh, anchors, anchor_threshold):
        # by IOU, anchor_threshold < 1
        box_wh = tf.expand_dims(gt_wh, 0)  # => 1 * n_gt * 2
        box_area = box_wh[..., 0] * box_wh[..., 1]  # => 1 * n_gt

        anchors = tf.cast(anchors, tf.float32)  # => n_anchor * 2
        anchors = tf.expand_dims(anchors, 1)  # => n_anchor * 1 * 2
        anchors_area = anchors[..., 0] * anchors[..., 1]  # => n_anchor * 1

        inter = tf.math.minimum(anchors[..., 0], box_wh[..., 0]) * tf.math.minimum(
            anchors[..., 1], box_wh[..., 1]
        )  # n_gt * n_anchor
        iou = inter / (anchors_area + box_area - inter + 1e-9)

        iou = iou > anchor_threshold
        return iou

    def enrich_pos_by_position(self, assigned_label, assigned_anchor, gain, matched_matrix, rect_style="rect4"):
        # using offset to extend more postive result, if x
        assigned_xy = assigned_label[..., 0:2]  # n_matched * 2
        offset = tf.constant([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], tf.float32)
        grid_offset = tf.zeros_like(assigned_xy)

        if rect_style == "rect2":
            g = 0.2  # offset
        elif rect_style == "rect4":
            g = 0.5  # offset
            matched = (assigned_xy % 1.0 < g) & (assigned_xy > 1.0)
            matched_left = matched[:, 0]
            matched_up = matched[:, 1]
            matched = (assigned_xy % 1.0 > (1 - g)) & (assigned_xy < tf.expand_dims(gain[0:2], 0) - 1.0)
            matched_right = matched[:, 0]
            matched_down = matched[:, 1]

            assigned_anchor = tf.concat(
                [
                    assigned_anchor,
                    assigned_anchor[matched_left],
                    assigned_anchor[matched_up],
                    assigned_anchor[matched_right],
                    assigned_anchor[matched_down],
                ],
                axis=0,
            )
            assigned_label = tf.concat(
                [
                    assigned_label,
                    assigned_label[matched_left],
                    assigned_label[matched_up],
                    assigned_label[matched_right],
                    assigned_label[matched_down],
                ],
                axis=0,
            )

            grid_offset = g * tf.concat(
                [
                    grid_offset,
                    grid_offset[matched_left] + offset[1],
                    grid_offset[matched_up] + offset[2],
                    grid_offset[matched_right] + offset[3],
                    grid_offset[matched_down] + offset[4],
                ],
                axis=0,
            )

        return assigned_label, assigned_anchor, grid_offset
