from configs.config import params
from model.optimizer import LrScheduler, Optimizer
import numpy as np
import tensorflow as tf

import tfyolo
from tfyolo import AutoConfig, AutoYolo, Trainer, YoloLoss
from tfyolo.datasets.load_data import DataLoader
from tfyolo.datasets.read_data import DataReader, transforms

np.random.seed(1919)
tf.random.set_seed(1949)


def run_train():
    trainer = Trainer(params)
    data_reader = DataReader(
        params["train_annotations_dir"],
        img_size=params["img_size"],
        transforms=transforms,
        mosaic=params["mosaic_data"],
        augment=params["augment_data"],
        filter_idx=None,
    )

    data_loader = DataLoader(
        data_reader,
        trainer.anchors,
        trainer.stride,
        params["img_size"],
        params["anchor_assign_method"],
        params["anchor_positive_augment"],
    )
    train_dataset = data_loader(batch_size=params["batch_size"], anchor_label=True)
    train_dataset.len = len(DataReader)

    trainer.train(train_dataset)


if __name__ == "__main__":
    run_train()
