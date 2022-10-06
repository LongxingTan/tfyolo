# tfyolo

[license-image]: https://img.shields.io/badge/license-Anti%20996-blue.svg
[license-url]: https://github.com/996icu/996.ICU/blob/master/LICENSE
[pypi-image]: https://badge.fury.io/py/tfts.svg
[pypi-url]: https://pypi.python.org/pypi/tfts
[build-image]: https://github.com/LongxingTan/Time-series-prediction/actions/workflows/test.yml/badge.svg?branch=master
[build-url]: https://github.com/LongxingTan/Time-series-prediction/actions/workflows/test.yml?query=branch%3Amaster
[docs-image]: https://readthedocs.org/projects/time-series-prediction/badge/?version=latest
[docs-url]: https://time-series-prediction.readthedocs.io/en/latest/

[![LICENSE][license-image]][license-url]
[![PyPI Version][pypi-image]][pypi-url]
[![Build Status][build-image]][build-url]
[![Docs Status][docs-image]][docs-url]

tfyolo is a YOLO (You only look once) library implemented by TensorFlow2 <br>

![demo](./data/sample/demo1.png)

## Key Features
- minimal Yolov5 by pure tensorflow2
- yaml file to configure the model
- custom data training
- mosaic data augmentation
- label encoding by iou or wh ratio of anchor
- positive sample augment
- multi-gpu training
- detailed code comments
- full of drawbacks with huge space to improve

## Tutorial
### prepare the data
```
$ bash data/scripts/get_voc.sh
$ cd yolo
$ python dataset/prepare_data.py
```

<!-- ### Download COCO
```
$ cd data/
$ bash get_coco_dataset.sh
``` -->

### Clone and install requirements
```
$ git clone git@github.com:LongxingTan/Yolov5.git
$ cd Yolov5/
$ pip install -r requirements.txt
```
<!-- ### Download pretrained weights
```
$ cd weights/
$ bash download_weights.sh
``` -->

### Train
```
$ python train.py
```


### Inference
```
$ python detect.py
$ python test.py
```

### Train on custom data
If you want to train on custom dataset, PLEASE note the input data should like this:
```
image_dir/001.jpg x_min, y_min, x_max, y_max, class_id x_min2, y_min2, x_max2, y_max2, class_id2
```
And maybe new anchor need to be created, don't forget to change the nc(number classes) in yolo-yaml.
```
$ python dataset/create_anchor.py
```

## Performance

| Model | Size | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | AP<sub>75</sub><sup>val</sup> |  cfg | weights |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: |
| YOLOV5s | 672 | 47.7% |52.6% | 61.4% | [cfg](https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/master/cfg/yolov4.cfg) | [weights](https://drive.google.com/file/d/137U-oLekAu-J-fe0E_seTblVxnU3tlNC/view?usp=sharing) |
| YOLOV5m | 672 | 47.7% |52.6% | 61.4% | [cfg](https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/master/cfg/yolov4.cfg) | [weights](https://drive.google.com/file/d/137U-oLekAu-J-fe0E_seTblVxnU3tlNC/view?usp=sharing) |
| YOLOV5l | 672 | 47.7% |52.6% | 61.4% | [cfg](https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/master/cfg/yolov4.cfg) | [weights](https://drive.google.com/file/d/137U-oLekAu-J-fe0E_seTblVxnU3tlNC/view?usp=sharing) |
| YOLOV5x | 672 | 47.7% |52.6% | 61.4% | [cfg](https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/master/cfg/yolov4.cfg) | [weights](https://drive.google.com/file/d/137U-oLekAu-J-fe0E_seTblVxnU3tlNC/view?usp=sharing) |
|  |  |  |  |  |  |  |


## Citation
If you find tf-yolo project useful in your research, please consider cite:
```
@misc{tfyolo2021,
    title={TFYOLO: yolo series benchmark in tensorflow},
    author={Longxing Tan},
    howpublished = {\url{https://github.com/longxingtan/tfyolo}},
    year={2021}
}
```
