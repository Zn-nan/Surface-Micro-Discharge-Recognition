"""
文件功能：提供 TFRecord 数据读取、解析、批处理和数据集划分工具。
代码分布：FEATURE_DESCRIPTION 定义样本格式；parse_tfrecord 解析单条样本；load_tfrecord 加载批数据；split_dataset 划分训练/验证/测试集。
整理思路：训练、评估、预览都应复用这里的解析逻辑，避免每个脚本复制一份 _parse_function。
使用方法：from smd.dataset import load_tfrecord，然后传入 TFRecord 路径即可得到 tf.data.Dataset。
"""

from pathlib import Path

import tensorflow as tf


FEATURE_DESCRIPTION = {
    "height": tf.io.FixedLenFeature([], tf.int64),
    "width": tf.io.FixedLenFeature([], tf.int64),
    "channel": tf.io.FixedLenFeature([], tf.int64),
    "label": tf.io.FixedLenFeature([], tf.int64),
    "image_raw": tf.io.FixedLenFeature([], tf.string),
}


def parse_tfrecord(example_proto, one_hot=True, num_classes=2, normalize=True):
    """解析单条 TFRecord 样本，返回图像张量和标签。"""
    features = tf.io.parse_single_example(example_proto, FEATURE_DESCRIPTION)
    image = tf.io.decode_raw(features["image_raw"], tf.uint8)
    image = tf.reshape(image, [features["height"], features["width"], features["channel"]])

    if normalize:
        image = tf.cast(image, tf.float32) / 255.0

    label = tf.cast(features["label"], tf.int32)
    if one_hot:
        label = tf.one_hot(label, depth=num_classes)

    return image, label


def read_tfrecord(path, one_hot=True, normalize=True):
    """读取 TFRecord 文件，返回尚未 batch 的 Dataset，便于后续 shuffle/split/cache。"""
    dataset = tf.data.TFRecordDataset(str(Path(path)))
    return dataset.map(lambda x: parse_tfrecord(x, one_hot=one_hot, normalize=normalize))


def load_tfrecord(path, batch_size=32, shuffle=False, one_hot=True):
    """加载 TFRecord 文件，并转换为可直接训练/评估的批数据。"""
    dataset = read_tfrecord(path, one_hot=one_hot)
    if shuffle:
        dataset = dataset.shuffle(10000, seed=42)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def split_dataset(dataset, train_size, val_size):
    """按样本数量切分训练集、验证集和测试集。"""
    train = dataset.take(train_size)
    val = dataset.skip(train_size).take(val_size)
    test = dataset.skip(train_size + val_size)
    return train, val, test

