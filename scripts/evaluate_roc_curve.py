# Converted from 20260626testmodels-shufflenet-ROCcurve.ipynb
# Edit paths before running on a new machine.

# %% Cell 1
'''
20260626  shufflenet
模型评估部分 测试集上测试模型
ROC曲线 混淆矩阵 AUC值等等
'''

# %% Cell 2
#%% ModelEvaluation 部分
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc,roc_auc_score
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import math

# %% Cell 3
## 读取 TFRecord 文件
# tfrecord_filename_train_Original = 'SMD_data/SMD_TFRecord/image_SMD_150_train_20241209.tfrecords'
tfrecord_filename_train_Augmented = 'SMD_data/SMD_TFRecord/image_SMD_150_train_20241209_Augmented.tfrecords'
tfrecord_filename_test = 'SMD_data/SMD_TFRecord/image_SMD_150_test_20241209.tfrecords'
batch_size = 32
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

# %% Cell 4
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dropout,Flatten,Dense,concatenate,Input
from tensorflow.keras.models import Model

from tensorflow.keras.layers import  AveragePooling2D
# 定义Inception 模块类
class Inception(tf.keras.layers.Layer):
    def __init__(self, cov1_1, cov_reduce3_3, cov3_3, cov_reduce5_5, cov5_5, pool_proj, **kwargs):
        super(Inception, self).__init__(**kwargs)
        self.branch1 = Conv2D(filters=cov1_1, kernel_size=1, activation='relu')

        self.branch2 = tf.keras.Sequential([
            Conv2D(filters=cov_reduce3_3, kernel_size=1, activation='relu'),
            Conv2D(filters=cov3_3, kernel_size=3, padding='same', activation='relu')
        ])

        self.branch3 = tf.keras.Sequential([
            Conv2D(filters=cov_reduce5_5, kernel_size=1, activation='relu'),
            Conv2D(filters=cov5_5, kernel_size=5, padding='same', activation='relu')
        ])

        self.branch4 = tf.keras.Sequential([
            MaxPool2D(pool_size=3, strides=1, padding='same'),
            Conv2D(filters=pool_proj, kernel_size=1, activation='relu')
        ])

    def call(self, inputs):
        branch1 = self.branch1(inputs)
        branch2 = self.branch2(inputs)
        branch3 = self.branch3(inputs)
        branch4 = self.branch4(inputs)
        outputs = concatenate([branch1, branch2, branch3, branch4])
        return outputs

    def get_config(self):
        config = super(Inception, self).get_config()
        config.update({
            'cov1_1': self.branch1.filters,
            'cov_reduce3_3': self.branch2.layers[0].filters,
            'cov3_3': self.branch2.layers[1].filters,
            'cov_reduce5_5': self.branch3.layers[0].filters,
            'cov5_5': self.branch3.layers[1].filters,
            'pool_proj': self.branch4.layers[1].filters
        })
        return config


# 定义GoogLeNet模型
def GoogLeNet(model_name='GoogLeNet', im_height=150, im_width=150, class_num=2):
    inputs = Input(shape=(im_height, im_width, 3), name='Inputs')

    x = Conv2D(filters=64, kernel_size=7, strides=2, padding='same', activation='relu', name='convolution-1')(inputs)
    x = MaxPool2D(pool_size=3, strides=2, padding='same', name='max-pool-1')(x)
    x = Conv2D(filters=192, kernel_size=3, strides=1, padding='same', activation='relu', name='convolution-2')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same', name='max-pool-2')(x)

    x = Inception(64, 96, 128, 16, 32, 32, name='inception-3a')(x)
    x = Inception(128, 128, 192, 32, 96, 64, name='inception-3b')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same', name='max-pool-3')(x)
    
    x = Inception(192, 96, 208, 16, 48, 64, name='inception-4a')(x)
    
    # 输出层
    x = AveragePooling2D(pool_size=(5, 5), strides=3)(x)
    x = Conv2D(128, kernel_size=1, activation='relu')(x)
    x = Flatten()(x)
    x = Dropout(0.7)(x)
    outputs = Dense(class_num, activation='softmax', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs, name=model_name)
    return model
def GoogLeNet_n(inception_layers=2, im_height=150, im_width=150, class_num=2):
    """
    创建GoogLeNet模型，inception_layers定义了模型中Inception模块的层数
    """
    inputs = Input(shape=(im_height, im_width, 3), name='Inputs')

    # 第一层卷积和池化
    x = Conv2D(filters=64, kernel_size=7, strides=2, padding='same', activation='relu', name='convolution-1')(inputs)
    x = MaxPool2D(pool_size=3, strides=2, padding='same', name='max-pool-1')(x)
    x = Conv2D(filters=192, kernel_size=3, strides=1, padding='same', activation='relu', name='convolution-2')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same', name='max-pool-2')(x)
    
    # Inception模块（根据传入的inception_layers参数来控制模块数）
    for i in range(inception_layers):
        x = Inception(64, 96, 128, 16, 32, 32, name=f'inception-{3+i}')(x)

    # 输出层
    x = AveragePooling2D(pool_size=(5, 5), strides=3)(x)
    x = Conv2D(128, kernel_size=1, activation='relu')(x)
    x = Flatten()(x)
    x = Dropout(0.7)(x)
    outputs = Dense(class_num, activation='softmax', name='output')(x)

    # 创建模型
    model = Model(inputs=inputs, outputs=outputs, name=f'GoogLeNet_{inception_layers}inception')
    return model

# %% Cell 5
from tensorflow.keras.layers import (Add, Conv2D, Dense, Dropout, Input,
                                     Lambda, Layer, Reshape, Softmax)
from tensorflow import keras
from tensorflow.keras import backend as K
#%%
#--------------------------------------#
#   LayerNormalization
#   层标准化的实现
#--------------------------------------#
class LayerNormalization(keras.layers.Layer):
    def __init__(self,
                 center=True,
                 scale=True,
                 epsilon=None,
                 gamma_initializer='ones',
                 beta_initializer='zeros',
                 gamma_regularizer=None,
                 beta_regularizer=None,
                 gamma_constraint=None,
                 beta_constraint=None,
                 **kwargs):
        """Layer normalization layer
        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
        :param center: Add an offset parameter if it is True.
        :param scale: Add a scale parameter if it is True.
        :param epsilon: Epsilon for calculating variance.
        :param gamma_initializer: Initializer for the gamma weight.
        :param beta_initializer: Initializer for the beta weight.
        :param gamma_regularizer: Optional regularizer for the gamma weight.
        :param beta_regularizer: Optional regularizer for the beta weight.
        :param gamma_constraint: Optional constraint for the gamma weight.
        :param beta_constraint: Optional constraint for the beta weight.
        :param kwargs:
        """
        super(LayerNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.center = center
        self.scale = scale
        if epsilon is None:
            epsilon = K.epsilon() * K.epsilon()
        self.epsilon = epsilon
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_initializer = keras.initializers.get(beta_initializer)
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
        self.beta_regularizer = keras.regularizers.get(beta_regularizer)
        self.gamma_constraint = keras.constraints.get(gamma_constraint)
        self.beta_constraint = keras.constraints.get(beta_constraint)
        self.gamma, self.beta = None, None

    def get_config(self):
        config = {
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'gamma_initializer': keras.initializers.serialize(self.gamma_initializer),
            'beta_initializer': keras.initializers.serialize(self.beta_initializer),
            'gamma_regularizer': keras.regularizers.serialize(self.gamma_regularizer),
            'beta_regularizer': keras.regularizers.serialize(self.beta_regularizer),
            'gamma_constraint': keras.constraints.serialize(self.gamma_constraint),
            'beta_constraint': keras.constraints.serialize(self.beta_constraint),
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask

    def build(self, input_shape):
        shape = input_shape[-1:]
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                name='gamma',
            )
        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                name='beta',
            )
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs, training=None):
        mean = K.mean(inputs, axis=-1, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        if self.scale:
            outputs *= self.gamma
        if self.center:
            outputs += self.beta
        return outputs

#--------------------------------------#
#   Gelu激活函数的实现
#   利用近似的数学公式
#--------------------------------------#
class Gelu(Layer):
    def __init__(self, **kwargs):
        super(Gelu, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return 0.5 * inputs * (1 + tf.tanh(tf.sqrt(2 / math.pi) * (inputs + 0.044715 * tf.pow(inputs, 3))))

    def get_config(self):
        config = super(Gelu, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

#--------------------------------------------------------------------------------------------------------------------#
#   classtoken部分是transformer的分类特征。用于堆叠到序列化后的图片特征中，作为一个单位的序列特征进行特征提取。
#
#   在利用步长为16x16的卷积将输入图片划分成14x14的部分后，将14x14部分的特征平铺，一幅图片会存在序列长度为196的特征。
#   此时生成一个classtoken，将classtoken堆叠到序列长度为196的特征上，获得一个序列长度为197的特征。
#   在特征提取的过程中，classtoken会与图片特征进行特征的交互。最终分类时，我们取出classtoken的特征，利用全连接分类。
#--------------------------------------------------------------------------------------------------------------------#
class ClassToken(Layer):
    def __init__(self, cls_initializer='zeros', cls_regularizer=None, cls_constraint=None, **kwargs):
        super(ClassToken, self).__init__(**kwargs)
        self.cls_initializer    = keras.initializers.get(cls_initializer)
        self.cls_regularizer    = keras.regularizers.get(cls_regularizer)
        self.cls_constraint     = keras.constraints.get(cls_constraint)

    def get_config(self):
        config = {
            'cls_initializer': keras.initializers.serialize(self.cls_initializer),
            'cls_regularizer': keras.regularizers.serialize(self.cls_regularizer),
            'cls_constraint': keras.constraints.serialize(self.cls_constraint),
        }
        base_config = super(ClassToken, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 1, input_shape[2])

    def build(self, input_shape):
        self.num_features = input_shape[-1]
        self.cls = self.add_weight(
            shape       = (1, 1, self.num_features),
            initializer = self.cls_initializer,
            regularizer = self.cls_regularizer,
            constraint  = self.cls_constraint,
            name        = 'cls',
        )
        super(ClassToken, self).build(input_shape)

    def call(self, inputs):
        batch_size      = tf.shape(inputs)[0]
        cls_broadcasted = tf.cast(tf.broadcast_to(self.cls, [batch_size, 1, self.num_features]), dtype = inputs.dtype)
        return tf.concat([cls_broadcasted, inputs], 1)

#--------------------------------------------------------------------------------------------------------------------#
#   为网络提取到的特征添加上位置信息。
#   以输入图片为224, 224, 3为例，我们获得的序列化后的图片特征为196, 768。加上classtoken后就是197, 768
#   此时生成的pos_Embedding的shape也为197, 768，代表每一个特征的位置信息。
#--------------------------------------------------------------------------------------------------------------------#
class AddPositionEmbs(Layer):
    def __init__(self, image_shape, patch_size, pe_initializer='zeros', pe_regularizer=None, pe_constraint=None, **kwargs):
        super(AddPositionEmbs, self).__init__(**kwargs)
        self.image_shape        = image_shape
        self.patch_size         = patch_size
        self.pe_initializer     = keras.initializers.get(pe_initializer)
        self.pe_regularizer     = keras.regularizers.get(pe_regularizer)
        self.pe_constraint      = keras.constraints.get(pe_constraint)

    def get_config(self):
        config = {
            'pe_initializer': keras.initializers.serialize(self.pe_initializer),
            'pe_regularizer': keras.regularizers.serialize(self.pe_regularizer),
            'pe_constraint': keras.constraints.serialize(self.pe_constraint),
        }
        base_config = super(AddPositionEmbs, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        assert (len(input_shape) == 3), f"Number of dimensions should be 3, got {len(input_shape)}"
        length  = (224 // self.patch_size) * (224 // self.patch_size) + 1
        self.pe = self.add_weight(
            # shape       = [1, input_shape[1], input_shape[2]],
            shape       = [1, length, input_shape[2]],
            initializer = self.pe_initializer,
            regularizer = self.pe_regularizer,
            constraint  = self.pe_constraint,
            name        = 'pos_embedding',
        )
        super(AddPositionEmbs, self).build(input_shape)

    def call(self, inputs):
        num_features = tf.shape(inputs)[2]

        cls_token_pe = self.pe[:, 0:1, :]
        img_token_pe = self.pe[:, 1: , :]

        img_token_pe = tf.reshape(img_token_pe, [1, (224 // self.patch_size), (224 // self.patch_size), num_features])
        img_token_pe = tf.compat.v1.image.resize_images(img_token_pe, (self.image_shape[0] // self.patch_size, self.image_shape[1] // self.patch_size), tf.image.ResizeMethod.BICUBIC, align_corners=False)
        img_token_pe = tf.reshape(img_token_pe, [1, -1, num_features])
        
        pe = tf.concat([cls_token_pe, img_token_pe], axis = 1)

        return inputs + tf.cast(pe, dtype=inputs.dtype)

#--------------------------------------------------------------------------------------------------------------------#
#   Attention机制
#   将输入的特征qkv特征进行划分，首先生成query, key, value。query是查询向量、key是键向量、v是值向量。
#   然后利用 查询向量query 叉乘 转置后的键向量key，这一步可以通俗的理解为，利用查询向量去查询序列的特征，获得序列每个部分的重要程度score。
#   然后利用 score 叉乘 value，这一步可以通俗的理解为，将序列每个部分的重要程度重新施加到序列的值上去。
#--------------------------------------------------------------------------------------------------------------------#
class Attention(Layer):
    def __init__(self, num_features, num_heads, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.num_features   = num_features
        self.num_heads      = num_heads
        self.projection_dim = num_features // num_heads

    def get_config(self):
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2] // 3)

    def call(self, inputs):
        #-----------------------------------------------#
        #   获得batch_size
        #-----------------------------------------------#
        bs      = tf.shape(inputs)[0]

        #-----------------------------------------------#
        #   b, 197, 3 * 768 -> b, 197, 3, 12, 64
        #-----------------------------------------------#
        inputs  = tf.reshape(inputs, [bs, -1, 3, self.num_heads, self.projection_dim])
        #-----------------------------------------------#
        #   b, 197, 3, 12, 64 -> 3, b, 12, 197, 64
        #-----------------------------------------------#
        inputs  = tf.transpose(inputs, [2, 0, 3, 1, 4])
        #-----------------------------------------------#
        #   将query, key, value划分开
        #   query     b, 12, 197, 64
        #   key       b, 12, 197, 64
        #   value     b, 12, 197, 64
        #-----------------------------------------------#
        query, key, value = inputs[0], inputs[1], inputs[2]
        #-----------------------------------------------#
        #   b, 12, 197, 64 @ b, 12, 197, 64 = b, 12, 197, 197
        #-----------------------------------------------#
        score           = tf.matmul(query, key, transpose_b=True)
        #-----------------------------------------------#
        #   进行数量级的缩放
        #-----------------------------------------------#
        scaled_score    = score / tf.math.sqrt(tf.cast(self.projection_dim, score.dtype))
        #-----------------------------------------------#
        #   b, 12, 197, 197 -> b, 12, 197, 197
        #-----------------------------------------------#
        weights         = tf.nn.softmax(scaled_score, axis=-1)
        #-----------------------------------------------#
        #   b, 12, 197, 197 @ b, 12, 197, 64 = b, 12, 197, 64
        #-----------------------------------------------#
        value          = tf.matmul(weights, value)

        #-----------------------------------------------#
        #   b, 12, 197, 64 -> b, 197, 12, 64
        #-----------------------------------------------#
        value = tf.transpose(value, perm=[0, 2, 1, 3])
        #-----------------------------------------------#
        #   b, 197, 12, 64 -> b, 197, 768
        #-----------------------------------------------#
        output = tf.reshape(value, (bs, -1, self.num_features))
        return output

def MultiHeadSelfAttention(inputs, num_features, num_heads, dropout, name):
    #-----------------------------------------------#
    #   qkv   b, 197, 768 -> b, 197, 3 * 768
    #-----------------------------------------------#
    qkv = Dense(int(num_features * 3), name = name + "qkv")(inputs)
    #-----------------------------------------------#
    #   b, 197, 3 * 768 -> b, 197, 768
    #-----------------------------------------------#
    x   = Attention(num_features, num_heads)(qkv)
    #-----------------------------------------------#
    #   197, 768 -> 197, 768
    #-----------------------------------------------#
    x   = Dense(num_features, name = name + "proj")(x)
    x   = Dropout(dropout)(x)
    return x

def MLP(y, num_features, mlp_dim, dropout, name):
    y = Dense(mlp_dim, name = name + "fc1")(y)
    y = Gelu()(y)
    y = Dropout(dropout)(y)
    y = Dense(num_features, name = name + "fc2")(y)
    return y

def TransformerBlock(inputs, num_features, num_heads, mlp_dim, dropout, name):
    #-----------------------------------------------#
    #   施加层标准化
    #-----------------------------------------------#
    x = LayerNormalization(epsilon=1e-6, name = name + "norm1")(inputs)
    #-----------------------------------------------#
    #   施加多头注意力机制
    #-----------------------------------------------#
    x = MultiHeadSelfAttention(x, num_features, num_heads, dropout, name = name + "attn.")
    x = Dropout(dropout)(x)
    #-----------------------------------------------#
    #   施加残差结构
    #-----------------------------------------------#
    x = Add()([x, inputs])

    #-----------------------------------------------#
    #   施加层标准化
    #-----------------------------------------------#
    y = LayerNormalization(epsilon=1e-6, name = name + "norm2")(x)
    #-----------------------------------------------#
    #   施加两次全连接
    #-----------------------------------------------#
    y = MLP(y, num_features, mlp_dim, dropout, name = name + "mlp.")
    y = Dropout(dropout)(y)
    #-----------------------------------------------#
    #   施加残差结构
    #-----------------------------------------------#
    y = Add()([x, y])
    return y

def VisionTransformer(input_shape = [224, 224], patch_size = 16, num_layers = 12, num_features = 768, num_heads = 12, mlp_dim = 3072, 
            classes = 1000, dropout = 0.1):
    #-----------------------------------------------#
    #   224, 224, 3
    #-----------------------------------------------#
    inputs      = Input(shape = (input_shape[0], input_shape[1], 3))
    
    #-----------------------------------------------#
    #   224, 224, 3 -> 14, 14, 768
    #-----------------------------------------------#
    x           = Conv2D(num_features, patch_size, strides = patch_size, padding = "valid", name = "patch_embed.proj")(inputs)
    #-----------------------------------------------#
    #   14, 14, 768 -> 196, 768
    #-----------------------------------------------#
    x           = Reshape(((input_shape[0] // patch_size) * (input_shape[1] // patch_size), num_features))(x)
    #-----------------------------------------------#
    #   196, 768 -> 197, 768
    #-----------------------------------------------#
    x           = ClassToken(name="cls_token")(x)
    #-----------------------------------------------#
    #   197, 768 -> 197, 768
    #-----------------------------------------------#
    x           = AddPositionEmbs(input_shape, patch_size, name="pos_embed")(x)
    #-----------------------------------------------#
    #   197, 768 -> 197, 768  12次
    #-----------------------------------------------#
    for n in range(num_layers):
        x = TransformerBlock(
            x,
            num_features= num_features,
            num_heads   = num_heads,
            mlp_dim     = mlp_dim,
            dropout     = dropout,
            name        = "blocks." + str(n) + ".",
        )
    x = LayerNormalization(
        epsilon=1e-6, name="norm"
    )(x)
    x = Lambda(lambda v: v[:, 0], name="ExtractToken")(x)
    x = Dense(classes, name="head")(x)
    x = Softmax()(x)
    return keras.models.Model(inputs, x)

# %% Cell 6
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

# %% Cell 7
#%% 加载以上训练的模型
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import time
# 模型文件的路径
model_names = ['ShuffleNetV2','LeNet','AlexNet','VGG','ResNet','MobileNet','ShuffleNet','SqueezeNet','ViT',
            'GoogLeNet-2','GoogLeNet-3','GoogLeNet-4','GoogLeNet-5','GoogLeNet-6',]
model_paths = []
for model_name in model_names:
    model_path = 'models/'+model_name+'_model_checkpoint_SMD_150_Augmented_20241212.h5'
# model_paths = ['models/GoogLeNet-2_model_checkpoint_SMD_150_Augmented_20241212.h5',
#               'models/GoogLeNet-3_model_checkpoint_SMD_150_Augmented_20241212.h5',
#               'models/GoogLeNet-4_model_checkpoint_SMD_150_Augmented_20241212.h5',
#               'models/GoogLeNet-5_model_checkpoint_SMD_150_Augmented_20241212.h5',
#               'models/GoogLeNet-6_model_checkpoint_SMD_150_Augmented_20241212.h5',
#               'models/MobileNet_model_checkpoint_SMD_150_Augmented_20241212.h5',
#               'models/ShuffleNet_model_checkpoint_SMD_150_Augmented_20241212.h5',
#               'models/SqueezeNet_model_checkpoint_SMD_150_Augmented_20241212.h5',
#               'models/ResNet_model_checkpoint_SMD_150_Augmented_20241212.h5',
#               'models/VGG_model_checkpoint_SMD_150_Augmented_20241212.h5',
#               'models/AlexNet_model_checkpoint_SMD_150_Augmented_20241212.h5',]


# 加载模型
# models = {}
models = []
#ShuffleNetV2_model = load_model('models/LeNet_model_checkpoint_SMD_150_Augmented_20241212.h5')
MODEL_PATH = 'models/ShuffleNetV2_model_checkpoint_SMD_150_Augmented_20260624-9936.h5'
tf.keras.backend.clear_session()
ShuffleNetV2_model = ShuffleNet(
    im_height=150,
    im_width=150,
    class_num=2,
    groups=2,
    width_multiplier=1.0
)
try:
    ShuffleNetV2_model.load_weights(MODEL_PATH)
except Exception as exc:
    print('load_weights failed, fallback to load_model:', exc)
    ShuffleNetV2_model = load_model(MODEL_PATH, compile=False)
learning_rate = 0.001
ShuffleNetV2_model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
ShuffleNetV2_model.summary()



LeNet_model = load_model('models/LeNet_model_checkpoint_SMD_150_Augmented_20241212.h5')

AlexNet_model = load_model('models/AlexNet_model_checkpoint_SMD_150_Augmented_20241212.h5')

# GoogLeNet_model = GoogLeNet(im_height=150, im_width=150, class_num=2)
# GoogLeNet_model.load_weights('GoogLeNet/GoogLeNet_model_checkpoint_SMD_150_AugmentedV2.h5')
# from tensorflow.keras.optimizers import Adam
# learning_rate=0.001
# GoogLeNet_model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

MobileNet_model = load_model('models/MobileNet_model_checkpoint_SMD_150_Augmented_20241212.h5')
ViT_model = VisionTransformer(input_shape = [150, 150], patch_size = 16, num_layers = 4, num_features = 256, num_heads = 4, mlp_dim = 512, 
            classes = 2, dropout = 0.1)
ViT_model.load_weights('models/ViT_model_checkpoint_SMD_150_Augmented_20241212.h5')
from tensorflow.keras.optimizers import Adam
learning_rate=0.001
ViT_model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

VGG_model = load_model('models/VGG_model_checkpoint_SMD_150_Augmented_20241212.h5')
ResNet_model = load_model('models/ResNet_model_checkpoint_SMD_150_Augmented_20241212.h5')
ShuffleNet_model = load_model('models/ShuffleNet_model_checkpoint_SMD_150_Augmented_20241212.h5')
SqueezeNet_model = load_model('models/SqueezeNet_model_checkpoint_SMD_150_Augmented_20241212.h5')
# AlexNet_model,VGG_model,ResNet_model,MobileNet_model,ShuffleNet_model,SqueezeNet_model,ViT_model,GoogLeNet_model
models.append(ShuffleNetV2_model)
models.append(LeNet_model)
models.append(AlexNet_model)
models.append(VGG_model)
models.append(ResNet_model)
models.append(MobileNet_model)
models.append(ShuffleNet_model)
models.append(SqueezeNet_model)
models.append(ViT_model)

GoogLeNet_model_paths = ['models/GoogLeNet-2_model_checkpoint_SMD_150_Augmented_20241212.h5',
              'models/GoogLeNet-3_model_checkpoint_SMD_150_Augmented_20241212.h5',
              'models/GoogLeNet-4_model_checkpoint_SMD_150_Augmented_20241212.h5',
              'models/GoogLeNet-5_model_checkpoint_SMD_150_Augmented_20241212.h5',
              'models/GoogLeNet-6_model_checkpoint_SMD_150_Augmented_20241212.h5',]
for i in range(len(GoogLeNet_model_paths)):
    GoogLeNet_model = GoogLeNet_n(inception_layers = 2+i,im_height=150, im_width=150, class_num=2)
    GoogLeNet_model.load_weights(GoogLeNet_model_paths[i])
    from tensorflow.keras.optimizers import Adam
    learning_rate=0.001
    GoogLeNet_model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    models.append(GoogLeNet_model)
    # starttime = time.time()
    # test_loss, test_accuracy = GoogLeNet_model.evaluate(test_dataset)
    # endtime = time.time()
    # print("inception layers:", i+2)
    # print("Test accuracy:", test_accuracy)
    # print((endtime-starttime)/12564)
print(len(models))

# %% Cell 8
for model in models:
    model.summary()

# %% Cell 9
#%% ModelEvaluation函数定义
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

import numpy as np
import tensorflow as tf

import numpy as np
import tensorflow as tf

class ModelEvaluation:
    def __init__(self, model, test_dataset):
        self.model = model
        self.test_dataset = test_dataset

        # 参数数量
        self.param_count = model.count_params()

        # Accuracy
        self.accuracy = self.calculate_accuracy()

        # 混淆矩阵
        self.confusion_matrix = self.calculate_confusion_matrix()

        # 混淆矩阵
        self.confusion_matrixV2 = self.calculate_confusion_matrixV2()
        
        # ROC曲线和AUC
        self.fpr, self.tpr, self.auc = self.calculate_roc_auc()

    def calculate_accuracy(self):
        correct_predictions = 0
        total_samples = 0

        for images, labels in self.test_dataset:
            predictions = self.model(images, training=False)
            predicted_labels = tf.argmax(predictions, axis=1)
            true_labels = tf.argmax(labels, axis=1)
            correct_predictions += tf.reduce_sum(tf.cast(predicted_labels == true_labels, tf.int32))
            total_samples += labels.shape[0]

        accuracy = correct_predictions / total_samples
        return accuracy.numpy()

    def calculate_confusion_matrix(self):
        #num_classes = self.test_dataset.element_spec[1].shape[1]
        num_classes = 2
        confusion = np.zeros((num_classes, num_classes), dtype=np.int32)

        for images, labels in self.test_dataset:
            predictions = self.model(images, training=False)
            predicted_labels = tf.argmax(predictions, axis=1)
            true_labels = tf.argmax(labels, axis=1)

            for true, predicted in zip(true_labels, predicted_labels):
                confusion[true, predicted] += 1

        return confusion
    def calculate_confusion_matrixV2(self):
        num_classes = len(np.unique(np.argmax(np.concatenate([labels.numpy() for _, labels in self.test_dataset]), axis=1)))
        confusion = np.zeros((num_classes, num_classes), dtype=np.int32)

        for images, labels in self.test_dataset:
            predictions = self.model(images, training=False)
            predicted_labels = tf.argmax(predictions, axis=1)
            true_labels = tf.argmax(labels, axis=1)

            for true, predicted in zip(true_labels, predicted_labels):
                confusion[true, predicted] += 1

        return confusion
        
    def calculate_roc_auc(self):
        y_true = []
        y_scores = []

        for images, labels in self.test_dataset:
            predictions = self.model(images, training=False)
            y_true.extend(labels.numpy())
            y_scores.extend(predictions.numpy())

        y_true = np.array(y_true)
        y_scores = np.array(y_scores)

        fpr, tpr, _ = roc_curve(y_true[:, 1], y_scores[:, 1])  # Assuming 1 is the positive class
        auc_value = auc(fpr, tpr)

        return fpr, tpr, auc_value
    def calculate_recall_precision(self):
        true_positives = np.diag(self.confusion_matrix)
        false_positives = np.sum(self.confusion_matrix, axis=0) - true_positives
        false_negatives = np.sum(self.confusion_matrix, axis=1) - true_positives

        recall = true_positives / (true_positives + false_negatives)
        precision = true_positives / (true_positives + false_positives)

        return recall, precision

    def print_recall_precision(self):
        recall, precision = self.calculate_recall_precision()
        for i in range(len(recall)):
            print(f'Class {i} - Recall: {recall[i]:.4f}, Precision: {precision[i]:.4f}')
    def calculate_f1_score(self):
        recall, precision = self.calculate_recall_precision()
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score

    def print_f1_score(self):
        f1_score = self.calculate_f1_score()
        for i in range(len(f1_score)):
            print(f'Class {i} - F1 Score: {f1_score[i]:.4f}') 
    def calculate_macro_average_f1_score(self):
        recall, precision = self.calculate_recall_precision()
        f1_score = 2 * (precision * recall) / (precision + recall)
        macro_average_f1 = np.mean(f1_score)
        return macro_average_f1

    def print_macro_average_f1_score(self):
        macro_average_f1 = self.calculate_macro_average_f1_score()
        print(f'宏平均F1分数：{macro_average_f1:.4f}')

    def plot_roc_curve(self):
        #plt.figure()
        plt.figure(figsize=(4, 4))
        plt.plot(self.fpr, self.tpr, color='deeppink', lw=4, label='ROC curve (area = {:.4f})'.format(self.auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=4, linestyle='--')
        # 设置 x 和 y 轴刻度的字体大小和字体类型
        #plt.xticks(fontsize=26, fontproperties='Times New Roman')
        #plt.yticks(fontsize=26, fontproperties='Times New Roman')
        
        
        plt.minorticks_on()
        plt.tick_params(axis="both", which="major", direction="out", width=1, length=5)
        plt.tick_params(axis="both", which="minor", direction="out", width=1, length=3)
        #plt.minorticks_on()
        plt.axis([0, 1, 0, 1])
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        # 获取当前 Axes 对象
        ax = plt.gca()
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        # plt.xticks(np.arange(0, 1.1, 0.1),  ["0", "", "0.2", "", "0.4", "", "0.6", "", "0.8", "", "1.0"], fontsize=24, fontproperties='Times New Roman')
        # #plt.minorticks_on()
        # plt.xticks(np.arange(0, 1.1, 0.1), ["" for _ in range(11)])
        
        # # 设置 y 轴大刻度和小刻度
        # plt.yticks(np.arange(0, 1.1, 0.1),  ["0", "", "0.2", "", "0.4", "", "0.6", "", "0.8", "", "1.0"], fontsize=24, fontproperties='Times New Roman')
        # plt.yticks(np.arange(0, 1.1, 0.1), ["" for _ in range(11)])
        
        # 设置刻度线粗细
        plt.tick_params(which='major', width=2, length=8)
        plt.tick_params(which='minor', width=2, length=4, labelsize=0)  # 不显示小刻度上的数值
        
        plt.grid(False)
        plt.xticks(fontsize=26,fontproperties = 'Times New Roman')
        plt.yticks(fontsize=26,fontproperties = 'Times New Roman')
        
        plt.xlabel('False Positive Rate', fontsize=28, fontproperties='Times New Roman')
        plt.ylabel('True Positive Rate', fontsize=28, fontproperties='Times New Roman')
        plt.title('ROC Curve', fontsize=30, fontproperties='Times New Roman')
        #plt.legend(loc='lower right', prop={'size': 26, 'family': 'Times New Roman'})
        plt.legend(loc='lower right', prop={'size': 26, 'family': 'Times New Roman'}, frameon=False)
    
        plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
# sns.set_theme(style="whitegrid",font='Times New Roman',font_scale=1.4)

# 绘制混淆矩阵
def plot_confusion_matrix(confusion_matrix, labels):
    plt.figure(figsize=(4, 3))
    # 指定字体
    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 26}
    #plt.rc('font', **font)
    ax = sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels,annot_kws={"fontsize":26,"fontproperties":'Times New Roman'})
    plt.xticks(fontsize=26,fontproperties = 'Times New Roman')
    plt.yticks(fontsize=26,fontproperties = 'Times New Roman')
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=26)
    plt.xlabel('Predicted',fontproperties = 'Times New Roman', size = 26)
    plt.ylabel('True',fontproperties = 'Times New Roman', size = 26)
    plt.title('Confusion Matrix',fontproperties = 'Times New Roman', size = 26)
    plt.show()

# %% Cell 10
#ViT
import time
import pandas as pd
model_names = ['ShuffleNetV2','LeNet','AlexNet','VGG','ResNet','MobileNet','ShuffleNet','SqueezeNet','ViT',
            'GoogLeNet-2','GoogLeNet-3','GoogLeNet-4','GoogLeNet-5','GoogLeNet-6',]
models_2 = [models[0]]
for i in range(len(models)):
    model = models[i]
    print(f'*****testing model {i} :{model_names[i]}*****')
    starttime = time.time()
    test_loss, test_accuracy = model.evaluate(test_dataset)
    endtime = time.time()
    print("Test loss:", test_loss)
    print("Test accuracy:", test_accuracy)
    print("Test time:",(endtime-starttime)/1571)
    
    evaluation = ModelEvaluation(model, test_dataset)
    print("Parameter Count:", evaluation.param_count)
    print("Accuracy:", evaluation.accuracy)
    print("Confusion Matrix:", evaluation.confusion_matrix)
    print("AUC:", evaluation.auc)

    evaluation.print_recall_precision() #Recall_Precision
    # Calculate and print F1-score
    evaluation.print_f1_score() #F1_score
    evaluation.print_macro_average_f1_score()
    #confusion_matrix
    plot_confusion_matrix(evaluation.confusion_matrix, labels=["Ozone", "Non-ozone"])
    #plot_confusion_matrix(GoogLeNet_evaluation.confusion_matrixV2, labels=["Ozone", "Non-ozone"])

    fpr, tpr, auc_value = evaluation.fpr, evaluation.tpr, evaluation.auc
    # print("False Positive Rate:", fpr)
    # print("True Positive Rate:", tpr)
    print("AUC Value:", auc_value) #AUC
    evaluation.plot_roc_curve() # 绘制 ROC 曲线
    df = pd.DataFrame(fpr)
    df.to_csv('excels/20260626'+model_names[i]+'fpr.csv', index=False)
    df = pd.DataFrame(tpr)
    df.to_csv('excels/20260626'+model_names[i]+'tpr.csv', index=False)
