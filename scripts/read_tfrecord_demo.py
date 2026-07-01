import tensorflow as tf
import matplotlib.pyplot as plt

# ===========================
# TFRecord 文件路径
# ===========================
tfrecord_path = r"SMD_data/SMD_TFRecord/image_SMD_150_train_20241209.tfrecords"

# ===========================
# 定义 TFRecord 格式
# ===========================
feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'channel': tf.io.FixedLenFeature([], tf.int64),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
}

# ===========================
# 解析单个样本
# ===========================
def parse_example(example_proto):

    example = tf.io.parse_single_example(
        example_proto,
        feature_description
    )

    height = tf.cast(example['height'], tf.int32)
    width = tf.cast(example['width'], tf.int32)
    channel = tf.cast(example['channel'], tf.int32)

    image = tf.io.decode_raw(
        example['image_raw'],
        tf.uint8
    )

    image = tf.reshape(
        image,
        (height, width, channel)
    )

    label = tf.cast(example['label'], tf.int32)

    return image, label


# ===========================
# 读取 TFRecord
# ===========================
dataset = tf.data.TFRecordDataset(tfrecord_path)
dataset = dataset.map(parse_example)

# ===========================
# 显示第一张图片
# ===========================
dataset = dataset.shuffle(1000)

for image, label in dataset.take(1):

    plt.figure(figsize=(5,5))
    plt.imshow(image.numpy())
    plt.title(f"Label = {label.numpy()}")
    plt.axis("off")
    plt.show()