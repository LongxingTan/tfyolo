[license-image]: https://img.shields.io/badge/license-Anti%20996-blue.svg
[license-url]: https://github.com/996icu/996.ICU/blob/master/LICENSE
[pypi-image]: https://badge.fury.io/py/tfyolo.svg
[pypi-url]: https://pypi.python.org/pypi/tfyolo
[pepy-image]: https://pepy.tech/badge/tfyolo/month
[pepy-url]: https://pepy.tech/project/tfyolo
[build-image]: https://github.com/LongxingTan/tf-yolo/actions/workflows/test.yml/badge.svg?branch=master
[build-url]: https://github.com/LongxingTan/tf-yolo/actions/workflows/test.yml?query=branch%3Amaster
[lint-image]: https://github.com/LongxingTan/tf-yolo/actions/workflows/lint.yml/badge.svg?branch=master
[lint-url]: https://github.com/LongxingTan/tf-yolo/actions/workflows/lint.yml?query=branch%3Amaster
[docs-image]: https://readthedocs.org/projects/tf-yolo/badge/?version=latest
[docs-url]: https://tf-yolo.readthedocs.io/en/latest/
[coverage-image]: https://codecov.io/gh/longxingtan/tf-yolo/branch/dev/graph/badge.svg
[coverage-url]: https://codecov.io/github/longxingtan/tf-yolo
[codeql-image]: https://github.com/longxingtan/tf-yolo/actions/workflows/codeql-analysis.yml/badge.svg
[codeql-url]: https://github.com/longxingtan/tf-yolo/actions/workflows/codeql-analysis.yml

<h1 align="center">
<img src="./docs/source/_static/logo.svg" width="400" align=center/>
</h1><br>

[![LICENSE][license-image]][license-url]
[![PyPI Version][pypi-image]][pypi-url]
[![Download][pepy-image]][pepy-url]
[![Build Status][build-image]][build-url]
[![Lint Status][lint-image]][lint-url]
[![Docs Status][docs-image]][docs-url]
[![Code Coverage][coverage-image]][coverage-url]
[![CodeQL Status][codeql-image]][codeql-url]

**[文档](https://tf-yolo.readthedocs.io)** | **[教程](https://tf-yolo.readthedocs.io/en/latest/tutorials.html)** | **[发布日志](https://tf-yolo.readthedocs.io/en/latest/CHANGELOG.html)** | **[English](https://github.com/LongxingTan/tf-yolo/blob/master/README.md)**

tfyolo是YOLO目标检测工具，采用TensorFlow框架。中文名：逆丑丑，来自"你瞅瞅"方言，引申自“你只瞅一次”。<br>
- 纯tensorflow2实现
- yaml文件配置模型
- 支持自定义数据训练
- 运用多种提升技巧
- 多GPU、TPU训练


![demo](examples/data/sample/demo1.png)


| Model | Size | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | AP<sub>75</sub><sup>val</sup> |  cfg | weights |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: |
| YOLOV5s | 672 | 47.7% |52.6% | 61.4% | [cfg](https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/master/cfg/yolov4.cfg) | [weights](https://drive.google.com/file/d/137U-oLekAu-J-fe0E_seTblVxnU3tlNC/view?usp=sharing) |
| YOLOV5m | 672 | 47.7% |52.6% | 61.4% | [cfg](https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/master/cfg/yolov4.cfg) | [weights](https://drive.google.com/file/d/137U-oLekAu-J-fe0E_seTblVxnU3tlNC/view?usp=sharing) |
| YOLOV5l | 672 | 47.7% |52.6% | 61.4% | [cfg](https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/master/cfg/yolov4.cfg) | [weights](https://drive.google.com/file/d/137U-oLekAu-J-fe0E_seTblVxnU3tlNC/view?usp=sharing) |
| YOLOV5x | 672 | 47.7% |52.6% | 61.4% | [cfg](https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/master/cfg/yolov4.cfg) | [weights](https://drive.google.com/file/d/137U-oLekAu-J-fe0E_seTblVxnU3tlNC/view?usp=sharing) |
|  |  |  |  |  |  |  |


## Usage

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
### Download VOC
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
