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
    features = tf.io.parse_single_example(example_proto, FEATURE_DESCRIPTION)
    image = tf.io.decode_raw(features["image_raw"], tf.uint8)
    image = tf.reshape(image, [features["height"], features["width"], features["channel"]])

    if normalize:
        image = tf.cast(image, tf.float32) / 255.0

    label = tf.cast(features["label"], tf.int32)
    if one_hot:
        label = tf.one_hot(label, depth=num_classes)

    return image, label


def load_tfrecord(path, batch_size=32, shuffle=False, one_hot=True):
    dataset = tf.data.TFRecordDataset(str(Path(path)))
    dataset = dataset.map(lambda x: parse_tfrecord(x, one_hot=one_hot))
    if shuffle:
        dataset = dataset.shuffle(10000, seed=42)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def split_dataset(dataset, train_size, val_size):
    train = dataset.take(train_size)
    val = dataset.skip(train_size).take(val_size)
    test = dataset.skip(train_size + val_size)
    return train, val, test

