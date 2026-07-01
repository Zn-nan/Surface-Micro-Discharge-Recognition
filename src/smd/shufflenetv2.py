"""
文件功能：定义用于表面微放电图像识别的轻量级 ShuffleNetV2 模型。
代码分布：ChannelShuffle 实现通道混洗；ShuffleUnitV2 实现基本单元；ShuffleNet 组装完整分类网络。
整理思路：模型结构只在这里维护一次，训练、量化、评估脚本都从本文件导入，避免结构不一致。
使用方法：from smd.shufflenetv2 import ShuffleNet，然后调用 ShuffleNet(im_height=150, im_width=150, class_num=2)。
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    DepthwiseConv2D,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
    ReLU,
)
from tensorflow.keras.models import Model


@tf.keras.utils.register_keras_serializable(package="Custom")
class ChannelShuffle(tf.keras.layers.Layer):
    """ShuffleNetV2 的通道混洗层，用于增强不同通道组之间的信息交换。"""

    def __init__(self, groups=2, **kwargs):
        super().__init__(**kwargs)
        self.groups = groups

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, height, width = input_shape[0], input_shape[1], input_shape[2]
        channels = inputs.shape[-1] or input_shape[3]
        channels_per_group = channels // self.groups

        x = tf.reshape(inputs, [batch_size, height, width, self.groups, channels_per_group])
        x = tf.transpose(x, [0, 1, 2, 4, 3])
        return tf.reshape(x, [batch_size, height, width, channels])

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
    prefix = f"{name}_" if name else ""
    x = Conv2D(
        filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
        name=prefix + "conv",
    )(x)
    x = BatchNormalization(name=prefix + "bn")(x)
    return ReLU(name=prefix + "relu")(x)


def dwconv_bn(x, kernel_size=3, strides=1, name=None):
    prefix = f"{name}_" if name else ""
    x = DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
        name=prefix + "dwconv",
    )(x)
    return BatchNormalization(name=prefix + "bn")(x)


def ShuffleUnitV2(x, out_channels, strides=1, name=None):
    """构建一个 ShuffleNetV2 基本单元。"""
    prefix = f"{name}_" if name else ""
    in_channels = int(x.shape[-1])

    if out_channels % 2 != 0:
        raise ValueError("out_channels must be divisible by 2 in ShuffleNet V2.")

    branch_channels = out_channels // 2

    if strides == 1:
        if in_channels != out_channels:
            raise ValueError("For strides=1, in_channels must equal out_channels.")

        x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)
        y = conv_bn_relu(x2, branch_channels, name=prefix + "branch2_pw1")
        y = dwconv_bn(y, name=prefix + "branch2_dw")
        y = conv_bn_relu(y, branch_channels, name=prefix + "branch2_pw2")
        out = Concatenate(axis=-1, name=prefix + "concat")([x1, y])
    elif strides == 2:
        x1 = dwconv_bn(x, strides=2, name=prefix + "branch1_dw")
        x1 = conv_bn_relu(x1, branch_channels, name=prefix + "branch1_pw")

        x2 = conv_bn_relu(x, branch_channels, name=prefix + "branch2_pw1")
        x2 = dwconv_bn(x2, strides=2, name=prefix + "branch2_dw")
        x2 = conv_bn_relu(x2, branch_channels, name=prefix + "branch2_pw2")
        out = Concatenate(axis=-1, name=prefix + "concat")([x1, x2])
    else:
        raise ValueError("strides must be 1 or 2.")

    return channel_shuffle(out, groups=2, name=prefix + "channel_shuffle")


def ShuffleNet(
    model_name="ShuffleNetV2_SMD_150",
    im_height=150,
    im_width=150,
    class_num=2,
    groups=2,
    width_multiplier=1,
):
    """构建适用于 150x150 SMD 图像二分类任务的 ShuffleNetV2 模型。"""
    def round_channels(ch):
        return max(int(round(ch / 2) * 2), 2)

    inputs = Input(shape=(im_height, im_width, 3), name="input")
    x = Conv2D(
        round_channels(24 * width_multiplier),
        (3, 3),
        strides=2,
        padding="same",
        use_bias=False,
        name="conv1",
    )(inputs)
    x = BatchNormalization(name="conv1_bn")(x)
    x = ReLU(name="conv1_relu")(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding="same", name="maxpool")(x)

    for stage, out_channels in enumerate(
        [round_channels(48 * width_multiplier), round_channels(96 * width_multiplier), round_channels(192 * width_multiplier)],
        start=2,
    ):
        x = ShuffleUnitV2(x, out_channels=out_channels, strides=2, name=f"stage{stage}_unit1")
        x = ShuffleUnitV2(x, out_channels=out_channels, strides=1, name=f"stage{stage}_unit2")
        x = ShuffleUnitV2(x, out_channels=out_channels, strides=1, name=f"stage{stage}_unit3")

    x = GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = Dropout(0.4, name="dropout")(x)
    outputs = Dense(class_num, activation="softmax", name="predictions")(x)
    return Model(inputs=inputs, outputs=outputs, name=model_name)

