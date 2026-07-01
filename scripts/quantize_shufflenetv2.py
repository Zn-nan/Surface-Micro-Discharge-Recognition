# Converted from 20260624_Model_quantization-shufflenetv2.ipynb
# Edit paths before running on a new machine.

# %% Cell 1
'''
20260624
????
ShuffleNet full-int8 ??
'''

# %% Cell 2
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
 
# 导入数据

import cv2 as cv
import os


import random
#from PIL import Image
from    tensorflow import keras
from    tensorflow.keras import optimizers,losses

tf.random.set_seed(2222)
np.random.seed(2222)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

#%%
def preprocess(x,y):

    # 转换成张量
    # x: [0,255]=> 0~1
    x = tf.cast(x, dtype=tf.float32) / 255.
    # 0~1 => D(0,1)
    #x = normalize(x) # 标准化
    y = tf.convert_to_tensor(y) # 转换成张量
    y = tf.one_hot(y, depth=2) #depth为模式种类数，与输出层相同
    return x, y



def load_img(root):
    # 创建数字编码表
    images = []
    labels = []
    # imgfiles_o3 = os.listdir(root +"/O3_original_Al2O3_ZrO2")
    imgfiles_o3 = [os.path.join(root+"/O3_original_Al2O3_ZrO2", filename) for filename in os.listdir(root +"/O3_original_Al2O3_ZrO2")
              if os.path.isfile(os.path.join(root+"/O3_original_Al2O3_ZrO2", filename)) and filename.lower().endswith('.jpg')]
    for file in imgfiles_o3:
        img_bgr =  cv.resize(cv.imread(file,1),(150,150))#resize图片像素
        img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
        images.append(img_rgb)
        labels.append(1)
        # print(1)

        # imgfiles_noo3 = os.listdir(root +"/NOO3_original_Al2O3_ZrO2")
    imgfiles_noo3 = [os.path.join(root+"/NOO3_original_Al2O3_ZrO2", filename) for filename in os.listdir(root +"/NOO3_original_Al2O3_ZrO2")
              if os.path.isfile(os.path.join(root+"/NOO3_original_Al2O3_ZrO2", filename)) and filename.lower().endswith('.jpg')]
    for file in imgfiles_noo3:
        img_bgr =  cv.resize(cv.imread(file,1),(150,150))#resize图片像素
        img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
        images.append(img_rgb)
        labels.append(0)
        # print(0)
 
    
    ##images labels按相同的顺序打乱 
    c = list(zip(images,labels))
    random.shuffle(c)
    images, labels = zip(*c)
    images = list(images)
    labels = list(labels)


    return images, labels



#%%
##读入数据


(x_train, y_train) = load_img('SMD_data/SMD_dataset')

x_train = np.array(x_train)
y_train = np.array(y_train)

y_train_onehot = keras.utils.to_categorical(y_train)
# 观察数据
#print (x_train.shape)
plt.imshow(x_train[1000])
print (y_train[1000])
 
train_images = x_train/255.0
y_train = y_train.astype('uint8')
y_train_onehot = y_train_onehot.astype('uint8')
#(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# 归一化
x_train = x_train / 255.0
 
class_names = ['NOO3','O3']
 
plt.imshow(x_train[1000])
 
 
x_train = x_train.astype('float32') 
#y_train = tf.one_hot(y_train, depth=2) #depth为模式种类数，与输出层相同

x_test = x_train[int(7852*0.8):]
y_test = y_train[int(7852*0.8):]
y_test_onehot = y_train_onehot[int(7852*0.8):]
x_train = x_train[:int(7852*0.8)]
y_train = y_train[:int(7852*0.8)]
y_train_onehot = y_train_onehot[:int(7852*0.8)]

# %% Cell 3
plt.imshow(x_train[1500])

# %% Cell 4
print(len(x_train))

# %% Cell 5
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

# %% Cell 6
#%% ??????? ShuffleNet ??
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

MODEL_PATH = 'models/ShuffleNetV2_model_checkpoint_SMD_150_Augmented_20260624-9936.h5'

# ??? 20241212ShuffleNet-train.ipynb ?????????????
# ??????? h5 ???????????????? load_model?
tf.keras.backend.clear_session()
ShuffleNet_model = ShuffleNet(
    im_height=150,
    im_width=150,
    class_num=2,
    groups=2,
    width_multiplier=1.0
)

try:
    ShuffleNet_model.load_weights(MODEL_PATH)
except Exception as exc:
    print('load_weights failed, fallback to load_model:', exc)
    ShuffleNet_model = load_model(MODEL_PATH, compile=False)

learning_rate = 0.001
ShuffleNet_model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
ShuffleNet_model.summary()

# %% Cell 7
train_images = x_train.astype('float32')
model = ShuffleNet_model

# full-int8 ??????? 20241217_Model_quantization.ipynb
TFLITE_FULL_INT_PATH = 'models/ShuffleNetV2_model-litemodel20260624.tflite'


def representative_data_gen():
    for image in train_images[:100]:
        yield [image.reshape(1, 150, 150, 3).astype('float32')]


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

# ??????????????
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# ????????? int8 ??
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model_quant = converter.convert()
open(TFLITE_FULL_INT_PATH, 'wb').write(tflite_model_quant)

# ????????
interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
print('input: ', input_details['dtype'], input_details['shape'], input_details['quantization'])
print('output: ', output_details['dtype'], output_details['shape'], output_details['quantization'])

class_names = ['NOO3', 'O3'] # NOO3:0 O3:1


#%% test
def run_tflite_model(tflite_file, test_image_indices):
    global x_test
    global y_test_onehot

    interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    predictions = np.zeros((len(test_image_indices),), dtype=int)
    for i, test_image_index in enumerate(test_image_indices):
        x = x_test[test_image_index]

        # ????????????? 0~1 float32 ????? int8/uint8?
        if input_details['dtype'] in (np.int8, np.uint8):
            input_scale, input_zero_point = input_details['quantization']
            x = x / input_scale + input_zero_point
            if input_details['dtype'] == np.int8:
                x = np.clip(x, -128, 127)
            else:
                x = np.clip(x, 0, 255)

        x = np.expand_dims(x, axis=0).astype(input_details['dtype'])
        interpreter.set_tensor(input_details['index'], x)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details['index'])[0]

        predictions[i] = output.argmax()

    return predictions


def evaluate_model(tflite_file, model_type):
    global x_test
    global y_test

    test_image_indices = range(x_test.shape[0])
    predictions = run_tflite_model(tflite_file, test_image_indices)
    num = 0
    for i in range(x_test.shape[0]):
        if y_test[i] == predictions[i]:
            num += 1
    accuracy = (num * 100) / len(x_test)

    print('%s model accuracy is %.4f%% (Number of test samples=%d)' % (
        model_type, accuracy, len(x_test)))
    return predictions


prediction = evaluate_model(TFLITE_FULL_INT_PATH, model_type='ShuffleNet full-int8')

# %% Cell 8
# ????????????
# ??????????? ShuffleNet full-int8 ???
# models/ShuffleNet_model-litemodel20260623.tflite

# %% Cell 9
# image_path = 'SMD_data/SMD_dataset/NOO3_original_Al2O3_ZrO2/SMD_2160.jpg'
image_path = 'SMD_9461.JPG'
class_names = ['NOO3', 'O3'] # NOO3:0 O3:1
model = ShuffleNet_model

a = cv.imread(image_path, 1)
img_bgr = cv.resize(cv.imread(image_path, 1), (150, 150)) # resize????
img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
img_array = tf.keras.preprocessing.image.img_to_array(img_rgb) / 255.0
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
class_names[np.argmax(predictions[0])]

# ????
from timeit import default_timer as timer
tic = timer()
# ??????1
img_bgr = cv.resize(cv.imread(image_path, 1), (150, 150)) # resize????
img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
img_array = tf.keras.preprocessing.image.img_to_array(img_rgb) / 255.0
img_array = tf.expand_dims(img_array, 0) # Create a batch
predictions = model.predict(img_array)
class_names[np.argmax(predictions[0])]
toc = timer()
time = toc - tic
print(time) # ??????????
# print(time/len(select_all_images))
