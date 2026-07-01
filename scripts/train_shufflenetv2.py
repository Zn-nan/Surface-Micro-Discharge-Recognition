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
# Shared ShuffleNetV2 model definition
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from smd_recognition.shufflenetv2 import ShuffleNet

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
