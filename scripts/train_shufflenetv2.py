# Converted from 20260624ShuffleNetv2-train.ipynb
# Edit paths before running on a new machine.

# %% Cell 1
#%% import
import os
import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# %% Cell 2
## 读取 TFRecord 文件
# tfrecord_filename_train_Original = 'SMD_data/SMD_TFRecord/image_SMD_150_train_20241209.tfrecords'
tfrecord_filename_train_Augmented = 'SMD_data/SMD_TFRecord/image_SMD_150_train_20241209_Augmented.tfrecords'
tfrecord_filename_test = 'SMD_data/SMD_TFRecord/image_SMD_150_test_20241209.tfrecords'
batch_size = 32*4
num_samples = 3926*2*8
train_size = int(3926*0.6*2*8)
val_size = int(3926*0.2*2*8)
# Create a dictionary describing the features.
def _parse_function(example_proto):

    feature_description = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'channel': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }
    features = tf.io.parse_single_example(example_proto, feature_description)
    #image = tf.io.decode_jpeg(features['image_raw'], channels=3)
    image_raw = tf.io.decode_raw(features["image_raw"], out_type=tf.uint8)
    image_raw = tf.cast(image_raw, tf.float32) 
    height = features["height"]
    width = features["width"]
    channel = features["channel"]
# 我们将数据转化为 bytes, 再转化为张量, 会转化为一个 1维数据
# 这里提前保存 shape 信息，转化回来
    image_raw = tf.reshape(image_raw, [height, width, channel])
    label = features["label"]
    image_raw = tf.cast(image_raw,tf.float32)/255.0
    label = tf.cast(label, tf.int32)
    label = tf.convert_to_tensor(label) # 转换成张量
    label = tf.one_hot(label, depth=2) #one-hot
    return image_raw, label
# raw_image_dataset = tf.data.TFRecordDataset(tfrecord_filename_train_Original)
# parsed_image_dataset = raw_image_dataset.map(_parse_function)
# train_val_dataset = parsed_image_dataset.shuffle(buffer_size=100000, seed=42) #buffer_size需要大于数据量
# #train_val_dataset = train_val_dataset.shuffle(buffer_size=(train_size+val_size), seed=42)
# train_dataset = train_val_dataset.take(train_size).cache()
# val_dataset = train_val_dataset.skip(train_size).take(val_size).cache()
raw_image_dataset = tf.data.TFRecordDataset(tfrecord_filename_train_Augmented)
parsed_image_dataset = raw_image_dataset.map(_parse_function)
train_val_dataset = parsed_image_dataset.shuffle(buffer_size=100000, seed=42) #buffer_size需要大于数据量
train_dataset = train_val_dataset.take(train_size).cache()
val_dataset = train_val_dataset.skip(train_size).take(val_size).cache()
raw_image_dataset = tf.data.TFRecordDataset(tfrecord_filename_test)
parsed_image_dataset = raw_image_dataset.map(_parse_function)
test_dataset = parsed_image_dataset.shuffle(buffer_size=100000, seed=42) #buffer_size需要大于数据量

train_dataset = train_dataset.batch(batch_size)
val_dataset = val_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)
# for image_features in train_dataset:
#     print(image_features)

# %% Cell 3
# ShuffleNet V2
# 适配输入: 150x150x3；输出: class_num=2
# 说明：本实现包含 ShuffleNet V2 的核心结构：
# channel split、1x1 conv、3x3 depthwise conv、concat、channel shuffle。

import tensorflow as tf
from tensorflow.keras.layers import (
    Concatenate, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Input, Dense,
    DepthwiseConv2D, BatchNormalization, ReLU, Dropout
)
from tensorflow.keras.models import Model


@tf.keras.utils.register_keras_serializable(package="Custom")
class ChannelShuffle(tf.keras.layers.Layer):
    """ShuffleNet V2 的通道混洗层。"""

    def __init__(self, groups=2, **kwargs):
        super().__init__(**kwargs)
        self.groups = groups

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, height, width = input_shape[0], input_shape[1], input_shape[2]
        channels = inputs.shape[-1]

        if channels is None:
            channels = input_shape[3]

        channels_per_group = channels // self.groups

        x = tf.reshape(
            inputs,
            [batch_size, height, width, self.groups, channels_per_group]
        )
        x = tf.transpose(x, [0, 1, 2, 4, 3])
        x = tf.reshape(x, [batch_size, height, width, channels])
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"groups": self.groups})
        return config


def channel_shuffle(x, groups=2, name=None):
    channels = x.shape[-1]
    if channels is not None and channels % groups != 0:
        raise ValueError(f"Channels ({channels}) must be divisible by groups ({groups}).")
    return ChannelShuffle(groups=groups, name=name)(x)


def conv_bn_relu(x, filters, kernel_size=1, strides=1, name=None):
    prefix = name + '_' if name else ''

    x = Conv2D(
        filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        use_bias=False,
        name=prefix + 'conv'
    )(x)
    x = BatchNormalization(name=prefix + 'bn')(x)
    x = ReLU(name=prefix + 'relu')(x)
    return x


def dwconv_bn(x, kernel_size=3, strides=1, name=None):
    prefix = name + '_' if name else ''

    x = DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        use_bias=False,
        name=prefix + 'dwconv'
    )(x)
    x = BatchNormalization(name=prefix + 'bn')(x)
    return x


def ShuffleUnitV2(x, out_channels, strides=1, name=None):
    """ShuffleNet V2 unit。

    strides=1:
        channel split -> branch2 conv -> concat -> channel shuffle

    strides=2:
        branch1: DWConv -> 1x1 Conv
        branch2: 1x1 Conv -> DWConv -> 1x1 Conv
        concat -> channel shuffle
    """
    prefix = name + '_' if name else ''
    in_channels = int(x.shape[-1])

    if out_channels % 2 != 0:
        raise ValueError("out_channels must be divisible by 2 in ShuffleNet V2.")

    branch_channels = out_channels // 2

    if strides == 1:
        if in_channels != out_channels:
            raise ValueError(
                "For strides=1 in ShuffleNet V2, in_channels must equal out_channels."
            )

        x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)

        y = conv_bn_relu(x2, branch_channels, kernel_size=1, strides=1,
                         name=prefix + 'branch2_pw1')
        y = dwconv_bn(y, kernel_size=3, strides=1,
                      name=prefix + 'branch2_dw')
        y = conv_bn_relu(y, branch_channels, kernel_size=1, strides=1,
                         name=prefix + 'branch2_pw2')

        out = Concatenate(axis=-1, name=prefix + 'concat')([x1, y])

    elif strides == 2:
        # branch1
        x1 = dwconv_bn(x, kernel_size=3, strides=2,
                       name=prefix + 'branch1_dw')
        x1 = conv_bn_relu(x1, branch_channels, kernel_size=1, strides=1,
                          name=prefix + 'branch1_pw')

        # branch2
        x2 = conv_bn_relu(x, branch_channels, kernel_size=1, strides=1,
                          name=prefix + 'branch2_pw1')
        x2 = dwconv_bn(x2, kernel_size=3, strides=2,
                       name=prefix + 'branch2_dw')
        x2 = conv_bn_relu(x2, branch_channels, kernel_size=1, strides=1,
                          name=prefix + 'branch2_pw2')

        out = Concatenate(axis=-1, name=prefix + 'concat')([x1, x2])

    else:
        raise ValueError("strides must be 1 or 2.")

    out = channel_shuffle(out, groups=2, name=prefix + 'channel_shuffle')
    return out


def ShuffleNet(model_name='ShuffleNetV2_SMD_150', im_height=150, im_width=150,
               class_num=2, groups=2, width_multiplier=1):
    input_shape = (im_height, im_width, 3)
    inputs = Input(shape=input_shape, name='input')

    def round_channels(ch):
        ch = int(round(ch / 2) * 2)
        return max(ch, 2)

    # 初始卷积
    init_channels = round_channels(24 * width_multiplier)

    x = Conv2D(
        init_channels,
        (3, 3),
        strides=2,
        padding='same',
        use_bias=False,
        name='conv1'
    )(inputs)
    x = BatchNormalization(name='conv1_bn')(x)
    x = ReLU(name='conv1_relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', name='maxpool')(x)

    # 各 stage 输出通道，保持和你原来的配置一致
    stage2_out = round_channels(48 * width_multiplier)
    stage3_out = round_channels(96 * width_multiplier)
    stage4_out = round_channels(192 * width_multiplier)

    # stage 2
    x = ShuffleUnitV2(x, out_channels=stage2_out, strides=2, name='stage2_unit1')
    x = ShuffleUnitV2(x, out_channels=stage2_out, strides=1, name='stage2_unit2')
    x = ShuffleUnitV2(x, out_channels=stage2_out, strides=1, name='stage2_unit3')

    # stage 3
    x = ShuffleUnitV2(x, out_channels=stage3_out, strides=2, name='stage3_unit1')
    x = ShuffleUnitV2(x, out_channels=stage3_out, strides=1, name='stage3_unit2')
    x = ShuffleUnitV2(x, out_channels=stage3_out, strides=1, name='stage3_unit3')

    # stage 4
    x = ShuffleUnitV2(x, out_channels=stage4_out, strides=2, name='stage4_unit1')
    x = ShuffleUnitV2(x, out_channels=stage4_out, strides=1, name='stage4_unit2')
    x = ShuffleUnitV2(x, out_channels=stage4_out, strides=1, name='stage4_unit3')

    x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = Dropout(0.4, name='dropout')(x)
    outputs = Dense(class_num, activation='softmax', name='predictions')(x)

    model = Model(inputs=inputs, outputs=outputs, name=model_name)
    return model

# %% Cell 4
# ShuffleNet V1 训练
model = ShuffleNet(
    im_height=150,
    im_width=150,
    class_num=2,
    groups=2,
    width_multiplier=1.0
)
model.summary()

# 保存网络结构表格，便于论文整理
import os
os.makedirs('models', exist_ok=True)
os.makedirs('excels', exist_ok=True)
os.makedirs('figures', exist_ok=True)

summary_lines = []
model.summary(print_fn=lambda x: summary_lines.append(x))
with open('models/ShuffleNetV2_model_summary_20260624.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(summary_lines))

# 编译模型
from tensorflow.keras.optimizers import Adam

learning_rate = 0.001
model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 保存模型 Checkpoint 回调
checkpoint_path = "models/ShuffleNetV2_model_checkpoint_SMD_150_Augmented_20260624.h5"

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    save_freq='epoch'
)

# 验证集 loss 连续 3 轮不下降，就把学习率减半
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=20,
    min_lr=5e-5,
    verbose=1
)

callbacks = [cp_callback, reduce_lr]

# 训练模型
import time
start_time = time.time()

history = model.fit(
    train_dataset,
    epochs=100,
    validation_data=val_dataset,
    callbacks=callbacks
)

end_time = time.time()
training_duration = end_time - start_time

training_duration_minutes = training_duration / 60
training_duration_hours = training_duration / 3600

print(f"Total training time: {training_duration} seconds")
print(f"Total training time: {training_duration_minutes} minutes")
print(f"Total training time: {training_duration_hours} hours")

# %% Cell 5
#%%
# 加载模型
from tensorflow.keras.models import load_model
model = load_model('models/ShuffleNetV2_model_checkpoint_SMD_150_Augmented_20260624.h5',
                   custom_objects={'ChannelShuffle': ChannelShuffle})
starttime = time.time()
test_loss, test_accuracy = model.evaluate(test_dataset)
endtime = time.time()

print("Test accuracy:", test_accuracy)
print("Test time:", (endtime - starttime) / 1570)

history = history.history

import pandas as pd
df = pd.DataFrame(history)
df.to_csv('excels/ShuffleNetV2-history-20260624.csv', index=False)

## 绘制 loss 和 accuracy 图像
plt.figure()
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('ShuffleNet V1 Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('figures/ShuffleNetV2-loss-20260624.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure()
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('ShuffleNet V1 Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('figures/ShuffleNetV2-accuracy-20260624.png', dpi=300, bbox_inches='tight')
plt.show()
