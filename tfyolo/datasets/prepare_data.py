#! /usr/bin/env python
# coding=utf-8
# @Author: Longxing Tan, tanlongxing888@163.com
# prepare the voc or coco data to a text for dataset/read_data.py with xyxy type
# because the script will add the information to existing file, so delete the txt file manually if run more times

import argparse
import os
import xml.etree.ElementTree as ET

from tqdm import tqdm


class VOCParser(object):
    def __init__(self, norm_bbox=False):
        """
        parse voc style xml data into txt, box coordinator normalize into (0,1) or keep pixel
        """
        self.norm_bbox = norm_bbox

    def parse(self, anno_file, data_base_dir, class_map, return_img=True):
        tree = ET.parse(anno_file)

        file_name = tree.findtext("filename")
        img_dir = os.path.join(data_base_dir, "JPEGImages", file_name)
        if return_img:
            img_dir = open(img_dir, "rb").read()

        height = float(tree.findtext("./size/height"))
        width = float(tree.findtext("./size/width"))
        xmin, ymin, xmax, ymax = [], [], [], []
        classes, classes_name = [], []

        for obj in tree.findall("object"):
            difficult = obj.find("difficult").text
            if difficult == "1":
                continue
            name = obj.find("name").text  # .encode('utf-8')
            bbox = obj.find("bndbox")
            xmin_ = float(bbox.find("xmin").text.strip())
            ymin_ = float(bbox.find("ymin").text.strip())
            xmax_ = float(bbox.find("xmax").text.strip())
            ymax_ = float(bbox.find("ymax").text.strip())
            if self.norm_bbox:
                xmin_ /= width
                ymin_ /= height
                xmax_ /= width
                ymax_ /= height
            classes_name.append(name)
            classes.append(class_map[name])

            xmin.append(xmin_)
            ymin.append(ymin_)
            xmax.append(xmax_)
            ymax.append(ymax_)
        return img_dir, xmin, ymin, xmax, ymax, classes, classes_name


class COCOParser(object):
    def __init__(self, norm_bbox=False):
        self.norm_bbox = norm_bbox

    def parse(self, anno_file):
        return


class DataPrepare(object):
    def __init__(self, data_dir, class_name_dir, output_dir, output_prefix="", data_style="voc"):
        if data_style == "voc":
            self.parser = VOCParser()
        elif data_style == "coco":
            self.parser = COCOParser()
        else:
            raise ValueError("only 'voc' and 'coco' are valid and supported data_style so far")

        self.xml_files = []
        for xml_file in os.listdir(os.path.join(data_dir, "Annotations")):
            self.xml_files.append(os.path.join(data_dir, "Annotations", xml_file))

        self.data_dir = data_dir
        self.output_dir = output_dir
        self.class_map = {name: idx for idx, name in enumerate(open(class_name_dir).read().splitlines())}
        self.output_prefix = output_prefix

    def write(self, split_weights=(0.9, 0.1, 0.0)):
        all_objects = self.get_objects()

        split1 = int(len(all_objects) * split_weights[0])
        split2 = int(len(all_objects) * (split_weights[0] + split_weights[1]))

        with open(self.output_dir + "/" + self.output_prefix + "train.txt", "w") as f:
            for objects in tqdm(all_objects[:split1]):
                self.write_single(f, objects)
        print("Train annotations generated, samples: {}".format(split1))

        with open(self.output_dir + "/" + self.output_prefix + "valid.txt", "w") as f:
            for objects in tqdm(all_objects[split1:split2]):
                self.write_single(f, objects)
        print("Valid annotations generated, samples: {}".format(split2 - split1))

        if split2 < 1:
            with open(self.output_dir + "/" + self.output_prefix + "test.txt", "w") as f:
                for objects in tqdm(all_objects[split2:]):
                    self.write_single(f, objects)
            print("Test annotations generated, samples: {}".format(len(all_objects) - split2))

    def write_single(self, f, objects):
        gt = [",".join([str(i[n_gt]) for i in objects[1:6]]) for n_gt in range(len(objects[1]))]
        objects_new = str(objects[0]) + " " + " ".join(gt)
        f.writelines(objects_new)
        f.writelines("\n")

    def get_objects(self):
        all_objects = []
        for xml in self.xml_files:
            objects = self.parser.parse(xml, self.data_dir, self.class_map, return_img=False)
            if objects is not None:
                all_objects.append(objects)
        # np.random.shuffle(all_objects)
        return all_objects


class VOCPrepare(object):
    def __init__(self, data_dir, class_name_dir, output_dir):
        self.parser = VOCParser()

        self.xml_files = []
        for xml_file in os.listdir(os.path.join(data_dir, "Annotations")):
            self.xml_files.append(os.path.join(data_dir, "Annotations", xml_file))

        self.data_dir = data_dir
        self.output_dir = output_dir
        self.class_map = {name: idx for idx, name in enumerate(open(class_name_dir).read().splitlines())}

    def write(self):
        all_objects = self.get_objects()

        with open(self.output_dir, "a") as f:
            for objects in tqdm(all_objects):
                self.write_single(f, objects)
        print("Text generated, samples: {}".format(len(all_objects)))

    def write_single(self, f, objects):
        gt = [",".join([str(i[n_gt]) for i in objects[1:6]]) for n_gt in range(len(objects[1]))]
        objects_new = str(objects[0]) + " " + " ".join(gt)
        f.writelines(objects_new)
        f.writelines("\n")

    def get_objects(self):
        all_objects = []
        for xml in self.xml_files:
            objects = self.parser.parse(xml, self.data_dir, self.class_map, return_img=False)
            if objects is not None:
                all_objects.append(objects)
        return all_objects


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=base_dir + "/data/voc", help="data directory")
    parser.add_argument("--class_name_dir", type=str, default=base_dir + "/data/voc/voc.names", help="class name dir")
    parser.add_argument("--output_dir", type=str, default=base_dir + "/data/voc", help="output text directory")
    opt = parser.parse_args()

    data_prepare = VOCPrepare(
        os.path.join(opt.data_dir, "train/VOCdevkit/VOC2007"),
        opt.class_name_dir,
        os.path.join(opt.output_dir, "voc_train.txt"),
    )
    data_prepare.write()

    data_prepare = VOCPrepare(
        os.path.join(opt.data_dir, "train/VOCdevkit/VOC2012"),
        opt.class_name_dir,
        os.path.join(opt.output_dir, "voc_train.txt"),
    )
    data_prepare.write()

    data_prepare = VOCPrepare(
        os.path.join(opt.data_dir, "test/VOCdevkit/VOC2007"),
        opt.class_name_dir,
        os.path.join(opt.output_dir, "voc_test.txt"),
    )
    data_prepare.write()
