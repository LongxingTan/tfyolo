#! /usr/bin/env python
# coding=utf-8
# @Author: Longxing Tan, tanlongxing888@163.com

import sys
import os
filePath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.split(filePath)[0])

import yaml
import shutil
import numpy as np
import tensorflow as tf
from configs.config import params
from dataset import DataReader
from dataset.image_utils import box_iou, xyxy2xywh, resize_back, resize_image
from model.metrics import ap_per_class
from model.post_process import batch_non_max_suppression


class TestDataReader(DataReader):
    def __init__(self, annotations_dir, image_target_size=640, transform=None, mosaic=False, augment=False):
        super(TestDataReader, self).__init__(annotations_dir, image_target_size, transform, mosaic, augment)

    def __getitem__(self, idx):
        img, label = self.load_image_and_label(idx)
        image_original_shape = img.shape[:2]
        img = resize_image(img, self.image_target_size, keep_ratio=True)
        img = img / 255.
        img_id = self.images_dir[idx].split('/')[-1].split('.')[0]
        return img_id, image_original_shape, img, label  # label is still original


class TestDataLoader(object):
    def __init__(self, data_reader):
        self.data_reader = data_reader

    def __call__(self, batch_size):
        dataset = tf.data.Dataset.from_generator(self.data_reader.iter,
                                                 output_types=(tf.string, tf.float32, tf.float32, tf.float32))
        dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        return dataset


class Evaluator(object):
    def __init__(self, model_dir, class_name_dir, img_size):
        self.model = tf.saved_model.load(model_dir)
        self.id2name = {idx: name for idx, name in enumerate(open(class_name_dir).read().splitlines())}
        self.num_classes = len(self.id2name)  # num_classes
        self.img_size = img_size

    def generate(self, test_dataset, results_dir):
        if os.path.exists(results_dir + '/predicted'):
            shutil.rmtree(results_dir + '/predicted')
        if os.path.exists(results_dir + '/ground-truth'):
            shutil.rmtree(results_dir + '/ground-truth')
        os.mkdir(results_dir + '/predicted')
        os.mkdir(results_dir + '/ground-truth')

        for image_id, image_original_shape, img, labels in test_dataset:

            predictions = self.model(img)
            predictions = [tf.reshape(x, (tf.shape(x)[0], -1, tf.shape(x)[-1])) for x in predictions]
            # batch_size * -1 * (num_class + 5)
            predictions = tf.concat(predictions, axis=1)
            # a list with n_image length, each element is n_pred * 6
            preds = batch_non_max_suppression(predictions, conf_threshold=0.4,
                                              iou_threshold=0.35)
            preds = [i.numpy() for i in preds]
            image_id = image_id.numpy().astype(str)
            labels = labels.numpy()

            for i, label in enumerate(labels):  # iter for image, gt
                with open(results_dir + '/ground-truth/{}.txt'.format(image_id[i]), 'w') as f:
                    for l in label:
                        class_name = self.id2name[l[4]]
                        xmin, ymin, xmax, ymax = list(map(str, l[:4]))

                        bbox_message = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
                        f.write(bbox_message)

            for i, (pred, original_shape) in enumerate(zip(preds, image_original_shape)):  # iter for image, pred
                if not pred.any():
                    with open(results_dir + '/predicted/{}.txt'.format(image_id[i]), 'w') as f:
                        f.write('')
                    break
                pred_bbox = pred[:, :4].astype(np.int32)
                pred_bbox = np.clip(pred_bbox, 0, self.img_size)
                pred_bbox = resize_back(pred_bbox, self.img_size, original_shape)
                pred_score = pred[:, 4].astype(np.float32)
                pred_class = pred[:, 5].astype(np.int32)

                with open(results_dir + '/predicted/{}.txt'.format(image_id[i]), 'w') as f:

                    for bbox, score, clss in zip(pred_bbox, pred_score, pred_class):
                        xmin, ymin, xmax, ymax = list(map(str, bbox))
                        score = '%.4f' % score
                        class_name = self.id2name[clss]

                        bbox_message = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'

                        f.write(bbox_message)
                        print('\t' + bbox_message)


if __name__ == '__main__':
    with open(params['yaml_dir']) as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
    anchors = np.array(yaml_dict['anchors'], np.float32).reshape(3, -1, 2)

    stride = np.array([8, 16, 32], np.float32)
    anchors = anchors / stride.reshape(-1, 1, 1)

    DataReader = TestDataReader(params['test_annotations_dir'])
    data_loader = TestDataLoader(DataReader)
    test_dataset = data_loader(batch_size=1)
    test_dataset.len = len(DataReader)

    evaluator = Evaluator(params['saved_model_dir'], params['class_name_dir'], img_size=params['img_size'])
    evaluator.generate(test_dataset, '../data/results')
