"""
文件功能：将表面微放电图像数据整理为 TFRecord 文件，并生成增强后的训练样本。
代码分布：前半部分读取原始图片和标签，中间定义图像增强方法，后半部分按训练集/测试集写入 TFRecord。
整理思路：脚本只作为数据准备入口；通用的 TFRecord 读取逻辑放在 src/smd/dataset.py 中，避免训练和评估重复解析代码。
使用方法：准备好本地 SMD_data 目录后，按实际路径修改本文件中的数据路径，然后运行 python scripts/make_tfrecords.py。
"""

# Converted from 20241209SMD_to_TFRecord_Augmented.ipynb
# Edit paths before running on a new machine.

# %% Cell 1
# -*- coding: utf-8 -*-
"""
Created on 2024 12 09
8个图
原图 平移旋转缩放×4 噪点 模糊 倾斜 
"""

#%% import
import tensorflow as tf
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# %% Cell 2
#%% 路径定义
## 设置图片路径和标签
train_data_dir = "SMD_data/SMD_dataset"
train_cats_dir = os.path.join(train_data_dir, "NOO3_original_Al2O3_ZrO2")
train_dogs_dir = os.path.join(train_data_dir, "O3_original_Al2O3_ZrO2")
# 过滤掉非图片文件（例如 .ipynb_checkpoints 文件夹）
train_cats = [os.path.join(train_cats_dir, filename) for filename in os.listdir(train_cats_dir)
              if os.path.isfile(os.path.join(train_cats_dir, filename)) and filename.lower().endswith('.jpg')]

train_dogs = [os.path.join(train_dogs_dir, filename) for filename in os.listdir(train_dogs_dir)
              if os.path.isfile(os.path.join(train_dogs_dir, filename)) and filename.lower().endswith('.jpg')]
# train_cats = [os.path.join(train_cats_dir, filename) for filename in os.listdir(train_cats_dir)]
# train_dogs = [os.path.join(train_dogs_dir, filename) for filename in os.listdir(train_dogs_dir)]
train_filenames = train_cats + train_dogs
train_labels = np.array([0]*len(train_cats) + [1]*len(train_dogs))
#SMD_data/SMD_dataset
#mycode_master/20231024SMD_deeplearning/SMD_data/SMD_dataset
#SMD_data/SMD_dataset/NOO3_original_Al2O3_ZrO2
#https://a244932-9f1d-07881bf0.westb.seetacloud.com:8443/jupyter/doc/tree/SMD_data/SMD_dataset
#NOO3 : 0 
#O3 : 1
#train_labels = tf.keras.utils.to_categorical(train_labels) #one-hot格式

# %% Cell 3
#%% 函数定义
## 函数定义
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))  
 
def image_example(image_string,height,width,channel,label):
  #image_shape = tf.image.decode_jpeg(image_string).shape

  feature = {
      'height': _int64_feature(height),
      'width': _int64_feature(width),
      'channel': _int64_feature(channel),
      'label': _int64_feature(label),
      'image_raw': _bytes_feature(image_string),
  }

  return tf.train.Example(features=tf.train.Features(feature=feature))

def scaling_and_center_image(original_image, scale_factor):
    if scale_factor <= 0:
        raise ValueError("Scale factor must be greater than 0.")
    
    original_height, original_width, _ = original_image.shape

    if scale_factor > 1:
        # Enlarge the image
        resized_image = cv.resize(original_image, None, fx=scale_factor, fy=scale_factor)
        resized_height, resized_width, _ = resized_image.shape

        # Calculate the center area boundaries
        center_x1 = (resized_width - original_width) // 2
        center_x2 = center_x1 + original_width
        center_y1 = (resized_height - original_height) // 2
        center_y2 = center_y1 + original_height

        # Crop the center area
        result_image = resized_image[center_y1:center_y2, center_x1:center_x2]
    else:
        # Reduce the image
        resized_image = cv.resize(original_image, None, fx=scale_factor, fy=scale_factor)
        resized_height, resized_width, _ = resized_image.shape
        # Create a black canvas with the original image dimensions
        canvas = np.zeros_like(original_image)
        # Calculate the offset to place the resized image in the center
        x_offset = (original_width - resized_width) // 2
        y_offset = (original_height - resized_height) // 2
        # Place the resized image in the center
        canvas[y_offset:y_offset + resized_height, x_offset:x_offset + resized_width] = resized_image
        result_image = canvas

    return result_image
# 数据增强函数
def augment_image(image):
    augmented_images = []
    deltax = 15 # 15/150 = 0.1
    deltaangle = 45 #旋转角度
    deltascale = 0.8
    deltanoise = 1 #噪声
    deltapts = 10 #透视变换
    rows, cols, _ = image.shape
        
    #原图
    augmented_images.append(image)
 

    # 旋转加缩放加平移
    angle = np.random.randint(-deltaangle, deltaangle)
    M_rotate = cv.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    augmented_image = cv.warpAffine(image, M_rotate, (cols, rows))
    scale = np.random.uniform(deltascale,1/deltascale)
    augmented_image = scaling_and_center_image(augmented_image,scale)
    tx = np.random.randint(-deltax, deltax)
    ty = np.random.randint(-deltax, deltax)
    M_translate = np.float32([[1, 0, tx], [0, 1, ty]])
    augmented_image = cv.warpAffine(augmented_image, M_translate, (cols, rows))
    augmented_images.append(augmented_image)
    
    # 旋转加缩放加平移
    angle = np.random.randint(-deltaangle, deltaangle)
    M_rotate = cv.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    augmented_image = cv.warpAffine(image, M_rotate, (cols, rows))
    scale = np.random.uniform(deltascale,1/deltascale)
    augmented_image = scaling_and_center_image(augmented_image,scale)
    tx = np.random.randint(-deltax, deltax)
    ty = np.random.randint(-deltax, deltax)
    M_translate = np.float32([[1, 0, tx], [0, 1, ty]])
    augmented_image = cv.warpAffine(augmented_image, M_translate, (cols, rows))
    augmented_images.append(augmented_image)
    
    # 旋转加缩放加平移
    angle = np.random.randint(-deltaangle, deltaangle)
    M_rotate = cv.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    augmented_image = cv.warpAffine(image, M_rotate, (cols, rows))
    scale = np.random.uniform(deltascale,1/deltascale)
    augmented_image = scaling_and_center_image(augmented_image,scale)
    tx = np.random.randint(-deltax, deltax)
    ty = np.random.randint(-deltax, deltax)
    M_translate = np.float32([[1, 0, tx], [0, 1, ty]])
    augmented_image = cv.warpAffine(augmented_image, M_translate, (cols, rows))
    augmented_images.append(augmented_image)
    
    # 旋转加缩放加平移
    angle = np.random.randint(-deltaangle, deltaangle)
    M_rotate = cv.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    augmented_image = cv.warpAffine(image, M_rotate, (cols, rows))
    scale = np.random.uniform(deltascale,1/deltascale)
    augmented_image = scaling_and_center_image(augmented_image,scale)
    tx = np.random.randint(-deltax, deltax)
    ty = np.random.randint(-deltax, deltax)
    M_translate = np.float32([[1, 0, tx], [0, 1, ty]])
    augmented_image = cv.warpAffine(augmented_image, M_translate, (cols, rows))
    augmented_images.append(augmented_image)
    
    # 添加噪点 旋转加缩放加平移
    angle = np.random.randint(-deltaangle, deltaangle)
    M_rotate = cv.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    augmented_image = cv.warpAffine(image, M_rotate, (cols, rows))
    scale = np.random.uniform(deltascale,1/deltascale)
    augmented_image = scaling_and_center_image(augmented_image,scale)
    tx = np.random.randint(-deltax, deltax)
    ty = np.random.randint(-deltax, deltax)
    M_translate = np.float32([[1, 0, tx], [0, 1, ty]])
    augmented_image = cv.warpAffine(augmented_image, M_translate, (cols, rows))
    
    noisy_image = np.copy(augmented_image)
    noise = np.random.normal(0, deltanoise, noisy_image.shape)
    noisy_image += noise.astype(np.uint8)
    augmented_images.append(noisy_image)

    
    # 随机模糊处理
    angle = np.random.randint(-deltaangle, deltaangle)
    M_rotate = cv.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    augmented_image = cv.warpAffine(image, M_rotate, (cols, rows))
    scale = np.random.uniform(deltascale,1/deltascale)
    augmented_image = scaling_and_center_image(augmented_image,scale)
    tx = np.random.randint(-deltax, deltax)
    ty = np.random.randint(-deltax, deltax)
    M_translate = np.float32([[1, 0, tx], [0, 1, ty]])
    augmented_image = cv.warpAffine(augmented_image, M_translate, (cols, rows))
    
    blur_type = np.random.choice(['gaussian', 'median', 'bilateral'])
    kernel_size = np.random.choice([3, 5])
    if blur_type == 'gaussian':
        blurred_image = cv.GaussianBlur(augmented_image, (kernel_size, kernel_size), 0)
    elif blur_type == 'median':
        blurred_image = cv.medianBlur(augmented_image, kernel_size)
    elif blur_type == 'bilateral':
        blurred_image = cv.bilateralFilter(augmented_image, 9, 75, 75)
    augmented_images.append(blurred_image)
    
    # 随机透视变换
    angle = np.random.randint(-deltaangle, deltaangle)
    M_rotate = cv.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    augmented_image = cv.warpAffine(image, M_rotate, (cols, rows))
    scale = np.random.uniform(deltascale,1/deltascale)
    augmented_image = scaling_and_center_image(augmented_image,scale)
    tx = np.random.randint(-deltax, deltax)
    ty = np.random.randint(-deltax, deltax)
    M_translate = np.float32([[1, 0, tx], [0, 1, ty]])
    augmented_image = cv.warpAffine(augmented_image, M_translate, (cols, rows))
    
    pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])
    noise = np.random.normal(0, deltapts, pts1.shape).astype(np.float32)  # Increase the standard deviation
    pts2 = pts1 + noise
    
    # Ensure that pts1 and pts2 have the correct shape
    pts1 = pts1.reshape(-1, 1, 2)
    pts2 = pts2.reshape(-1, 1, 2)
    
    M_perspective = cv.getPerspectiveTransform(pts1, pts2)
    perspective_image = cv.warpPerspective(augmented_image, M_perspective, (cols, rows))
    augmented_images.append(perspective_image)

    return augmented_images

# %% Cell 4
# 随机打乱 
import random
zipped = list(zip(train_filenames, train_labels))  # 将 zip 对象转换为列表
random.shuffle(zipped)  # 打乱列表中的元素
# 计算训练集和测试集的大小
train_size = int(0.8 * len(zipped))  # 80% 的数据用于训练集
test_size = len(zipped) - train_size  # 剩余的 20% 用于测试集
# 划分训练集和测试集
train_data = zipped[:train_size]
test_data = zipped[train_size:]
# # 如果需要重新分开 train_filenames 和 train_labels：
# train_filenames, train_labels = zip(*zipped)  # 解压打乱后的元组
# train_filenames = list(train_filenames)  # 如果需要将结果转换回列表
# train_labels = list(train_labels)
#print(train_labels)
#print(zip(train_filenames, train_labels))

# %% Cell 5
#%% 写入TFRecord文件 训练集 未增强
# 创建一个计数器，并在每次添加样本时递增
sample_counter = 0
record_file = 'SMD_data/SMD_TFRecord/image_SMD_150_train_20241209.tfrecords'
#record_file = 'SMD_data/SMD_TFRecord/image_SMD_150.tfrecords'
#SMD_data/SMD_TFRecord
# record_file = 'E:/SMD_data/SMD_TFRecord/image_SMD_150_Augmented.tfrecords'
data = train_data
# 创建 TFRecord 文件
with tf.io.TFRecordWriter(record_file) as writer:
    num_images = len(data)
    batch_size = 50
    processed_images = 0
    for i, (filename, label) in enumerate(data, start=1):
        sample_counter += 1
        img_bgr = cv.imread(filename, 1)
        img_bgr = cv.resize(img_bgr, (150, 150))
        img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

        # 执行多个数据增强操作
        augmented_images = [img_rgb]

        for augmented_image in augmented_images:
            height = augmented_image.shape[0]
            width = augmented_image.shape[1]
            channel = augmented_image.shape[2]
            image_bytes = augmented_image.tobytes()
            tf_example = image_example(image_bytes, height, width, channel, label)
            writer.write(tf_example.SerializeToString())
        processed_images += 1

        # 输出进度更新
        if processed_images % batch_size == 0:
            if processed_images % (batch_size * 10) == 0:
                print(f'-{processed_images} / {len(train_filenames)}')  # 输出一行连字符
            else:
                print('-', end='')
    # 所有图像处理完成后输出最终进度
    print(f'Processed {processed_images} out of {num_images} images')
    # feature = {
    #     'num_samples': _int64_feature(sample_counter),
    # }

    # tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    # writer.write(tf_example.SerializeToString())
    # print('end!')

# with tf.io.TFRecordWriter(record_file) as writer:
#     feature = {
#         'num_samples': _int64_feature(sample_counter),
#     }

#     tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
#     writer.write(tf_example.SerializeToString())

# %% Cell 7
print(filename)
print(label)
print(i)
print(train_data[3934])
print(train_data[3933])

# %% Cell 8
#%% 写入TFRecord文件 训练集 已增强
# 创建一个计数器，并在每次添加样本时递增
sample_counter = 0
record_file = 'SMD_data/SMD_TFRecord/image_SMD_150_train_20241209_Augmented.tfrecords'
#record_file = 'SMD_data/SMD_TFRecord/image_SMD_150.tfrecords'
#SMD_data/SMD_TFRecord
# record_file = 'E:/SMD_data/SMD_TFRecord/image_SMD_150_Augmented.tfrecords'
data = train_data
# 创建 TFRecord 文件
with tf.io.TFRecordWriter(record_file) as writer:
    num_images = len(data)
    batch_size = 50
    processed_images = 0
    for i, (filename, label) in enumerate(data, start=1):
        sample_counter += 1
        img_bgr = cv.imread(filename, 1)
        img_bgr = cv.resize(img_bgr, (150, 150))
        img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

        # 执行多个数据增强操作
        augmented_images = augment_image(img_rgb)

        for augmented_image in augmented_images:
            height = augmented_image.shape[0]
            width = augmented_image.shape[1]
            channel = augmented_image.shape[2]
            image_bytes = augmented_image.tobytes()
            tf_example = image_example(image_bytes, height, width, channel, label)
            writer.write(tf_example.SerializeToString())
        processed_images += 1

        # 输出进度更新
        if processed_images % batch_size == 0:
            if processed_images % (batch_size * 10) == 0:
                print(f'-{processed_images} / {len(train_filenames)}')  # 输出一行连字符
            else:
                print('-', end='')
    # 所有图像处理完成后输出最终进度
    print(f'Processed {processed_images} out of {num_images} images')
    # feature = {
    #     'num_samples': _int64_feature(sample_counter),
    # }

    # tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    # writer.write(tf_example.SerializeToString())
    # print('end!')

# with tf.io.TFRecordWriter(record_file) as writer:
#     feature = {
#         'num_samples': _int64_feature(sample_counter),
#     }

#     tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
#     writer.write(tf_example.SerializeToString())

# %% Cell 9
#%% 写入TFRecord文件 测试集 未增强
# 创建一个计数器，并在每次添加样本时递增
sample_counter = 0
record_file = 'SMD_data/SMD_TFRecord/image_SMD_150_test_20241209.tfrecords'
#record_file = 'SMD_data/SMD_TFRecord/image_SMD_150.tfrecords'
#SMD_data/SMD_TFRecord
# record_file = 'E:/SMD_data/SMD_TFRecord/image_SMD_150_Augmented.tfrecords'
data = test_data
# 创建 TFRecord 文件
with tf.io.TFRecordWriter(record_file) as writer:
    num_images = len(data)
    batch_size = 50
    processed_images = 0
    for i, (filename, label) in enumerate(data, start=1):
        sample_counter += 1
        img_bgr = cv.imread(filename, 1)
        img_bgr = cv.resize(img_bgr, (150, 150))
        img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

        # 执行多个数据增强操作
        augmented_images = [img_rgb]

        for augmented_image in augmented_images:
            height = augmented_image.shape[0]
            width = augmented_image.shape[1]
            channel = augmented_image.shape[2]
            image_bytes = augmented_image.tobytes()
            tf_example = image_example(image_bytes, height, width, channel, label)
            writer.write(tf_example.SerializeToString())
        processed_images += 1

        # 输出进度更新
        if processed_images % batch_size == 0:
            if processed_images % (batch_size * 10) == 0:
                print(f'-{processed_images} / {len(train_filenames)}')  # 输出一行连字符
            else:
                print('-', end='')
    # 所有图像处理完成后输出最终进度
    print(f'Processed {processed_images} out of {num_images} images')
    # feature = {
    #     'num_samples': _int64_feature(sample_counter),
    # }

    # tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    # writer.write(tf_example.SerializeToString())
    # print('end!')

# with tf.io.TFRecordWriter(record_file) as writer:
#     feature = {
#         'num_samples': _int64_feature(sample_counter),
#     }

#     tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
#     writer.write(tf_example.SerializeToString())

# %% Cell 10
#%% 写入TFRecord文件 测试集 已增强
# 创建一个计数器，并在每次添加样本时递增
sample_counter = 0
record_file = 'SMD_data/SMD_TFRecord/image_SMD_150_11_traintest_20241209_Augmented.tfrecords'
#record_file = 'SMD_data/SMD_TFRecord/image_SMD_150.tfrecords'
#SMD_data/SMD_TFRecord
# record_file = 'E:/SMD_data/SMD_TFRecord/image_SMD_150_Augmented.tfrecords'
data = zipped
# 创建 TFRecord 文件
with tf.io.TFRecordWriter(record_file) as writer:
    num_images = len(data)
    batch_size = 50
    processed_images = 0
    for i, (filename, label) in enumerate(data, start=1):
        sample_counter += 1
        img_bgr = cv.imread(filename, 1)
        img_bgr = cv.resize(img_bgr, (150, 150))
        img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

        # 执行多个数据增强操作
        augmented_images = augment_image(img_rgb)

        for augmented_image in augmented_images:
            height = augmented_image.shape[0]
            width = augmented_image.shape[1]
            channel = augmented_image.shape[2]
            image_bytes = augmented_image.tobytes()
            tf_example = image_example(image_bytes, height, width, channel, label)
            writer.write(tf_example.SerializeToString())
        processed_images += 1

        # 输出进度更新
        if processed_images % batch_size == 0:
            if processed_images % (batch_size * 10) == 0:
                print(f'-{processed_images} / {len(train_filenames)}')  # 输出一行连字符
            else:
                print('-', end='')
    # 所有图像处理完成后输出最终进度
    print(f'Processed {processed_images} out of {num_images} images')
    # feature = {
    #     'num_samples': _int64_feature(sample_counter),
    # }

    # tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    # writer.write(tf_example.SerializeToString())
    # print('end!')

# with tf.io.TFRecordWriter(record_file) as writer:
#     feature = {
#         'num_samples': _int64_feature(sample_counter),
#     }

#     tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
#     writer.write(tf_example.SerializeToString())

# %% Cell 11
# train_filenames = train_cats + train_dogs
# train_labels = np.array([0]*len(train_cats) + [1]*len(train_dogs))
# 确保两个列表的长度相等，选择相同数量的数据
min_len = min(len(train_cats), len(train_dogs))  # 取最小长度
train_cats_1_1 = train_cats[:min_len]
train_dogs_1_1 = train_dogs[:min_len]
# 合并数据
train_filenames_1_1 = train_cats_1_1 + train_dogs_1_1
train_labels_1_1 = np.array([0]*len(train_cats_1_1) + [1]*len(train_dogs_1_1))

# 确保 train_cats 的数量是 train_dogs 的 5 倍
train_cats_5_1 = train_cats
train_dogs_5_1 = train_dogs[:int(0.2 * min_len)]  # 使用所有的 train_dogs
# 合并数据
train_filenames_5_1 = train_cats_5_1 + train_dogs_5_1
train_labels_5_1 = np.array([0]*len(train_cats_5_1) + [1]*len(train_dogs_5_1))

# 确保 train_dogs 的数量是 train_cats 的 5 倍
train_dogs_1_5 = train_dogs
train_cats_1_5 = train_cats[:int(0.2 * min_len)]   # 使用所有的 train_cats
# 合并数据
train_filenames_1_5 = train_cats_1_5 + train_dogs_1_5
train_labels_1_5 = np.array([0]*len(train_cats_1_5) + [1]*len(train_dogs_1_5))

# 随机打乱 
import random
data_1_1 = list(zip(train_filenames_1_1, train_labels_1_1))  # 将 zip 对象转换为列表
random.shuffle(data_1_1)  # 打乱列表中的元素
data_5_1 = list(zip(train_filenames_5_1, train_labels_5_1))  # 将 zip 对象转换为列表
random.shuffle(data_5_1)  # 打乱列表中的元素
data_1_5 = list(zip(train_filenames_1_5, train_labels_1_5))  # 将 zip 对象转换为列表
random.shuffle(data_1_5)  # 打乱列表中的元素
#print(data_1_1)

# %% Cell 12
#%% 写入TFRecord文件 测试集 已增强
# 创建一个计数器，并在每次添加样本时递增
sample_counter = 0
record_file = 'SMD_data/SMD_TFRecord/image_SMD_150_1_1_traintest_20241209_Augmented.tfrecords'
#record_file = 'SMD_data/SMD_TFRecord/image_SMD_150.tfrecords'
#SMD_data/SMD_TFRecord
# record_file = 'E:/SMD_data/SMD_TFRecord/image_SMD_150_Augmented.tfrecords'
data = data_1_1
# 创建 TFRecord 文件
with tf.io.TFRecordWriter(record_file) as writer:
    num_images = len(data)
    batch_size = 50
    processed_images = 0
    for i, (filename, label) in enumerate(data, start=1):
        sample_counter += 1
        img_bgr = cv.imread(filename, 1)
        img_bgr = cv.resize(img_bgr, (150, 150))
        img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

        # 执行多个数据增强操作
        augmented_images = augment_image(img_rgb)

        for augmented_image in augmented_images:
            height = augmented_image.shape[0]
            width = augmented_image.shape[1]
            channel = augmented_image.shape[2]
            image_bytes = augmented_image.tobytes()
            tf_example = image_example(image_bytes, height, width, channel, label)
            writer.write(tf_example.SerializeToString())
        processed_images += 1

        # 输出进度更新
        if processed_images % batch_size == 0:
            if processed_images % (batch_size * 10) == 0:
                print(f'-{processed_images} / {len(train_filenames)}')  # 输出一行连字符
            else:
                print('-', end='')
    # 所有图像处理完成后输出最终进度
    print(f'Processed {processed_images} out of {num_images} images')
    # feature = {
    #     'num_samples': _int64_feature(sample_counter),
    # }

    # tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    # writer.write(tf_example.SerializeToString())
    # print('end!')

# with tf.io.TFRecordWriter(record_file) as writer:
#     feature = {
#         'num_samples': _int64_feature(sample_counter),
#     }

#     tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
#     writer.write(tf_example.SerializeToString())

# %% Cell 13
#%% 写入TFRecord文件 测试集 已增强
# 创建一个计数器，并在每次添加样本时递增
sample_counter = 0
record_file = 'SMD_data/SMD_TFRecord/image_SMD_150_5_1_traintest_20241209_Augmented.tfrecords'
#record_file = 'SMD_data/SMD_TFRecord/image_SMD_150.tfrecords'
#SMD_data/SMD_TFRecord
# record_file = 'E:/SMD_data/SMD_TFRecord/image_SMD_150_Augmented.tfrecords'
data = data_5_1
# 创建 TFRecord 文件
with tf.io.TFRecordWriter(record_file) as writer:
    num_images = len(data)
    batch_size = 50
    processed_images = 0
    for i, (filename, label) in enumerate(data, start=1):
        sample_counter += 1
        img_bgr = cv.imread(filename, 1)
        img_bgr = cv.resize(img_bgr, (150, 150))
        img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

        # 执行多个数据增强操作
        augmented_images = augment_image(img_rgb)

        for augmented_image in augmented_images:
            height = augmented_image.shape[0]
            width = augmented_image.shape[1]
            channel = augmented_image.shape[2]
            image_bytes = augmented_image.tobytes()
            tf_example = image_example(image_bytes, height, width, channel, label)
            writer.write(tf_example.SerializeToString())
        processed_images += 1

        # 输出进度更新
        if processed_images % batch_size == 0:
            if processed_images % (batch_size * 10) == 0:
                print(f'-{processed_images} / {len(train_filenames)}')  # 输出一行连字符
            else:
                print('-', end='')
    # 所有图像处理完成后输出最终进度
    print(f'Processed {processed_images} out of {num_images} images')
    # feature = {
    #     'num_samples': _int64_feature(sample_counter),
    # }

    # tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    # writer.write(tf_example.SerializeToString())
    # print('end!')

# with tf.io.TFRecordWriter(record_file) as writer:
#     feature = {
#         'num_samples': _int64_feature(sample_counter),
#     }

#     tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
#     writer.write(tf_example.SerializeToString())

# %% Cell 14
#%% 写入TFRecord文件 测试集 已增强
# 创建一个计数器，并在每次添加样本时递增
sample_counter = 0
record_file = 'SMD_data/SMD_TFRecord/image_SMD_150_1_5_traintest_20241209_Augmented.tfrecords'
#record_file = 'SMD_data/SMD_TFRecord/image_SMD_150.tfrecords'
#SMD_data/SMD_TFRecord
# record_file = 'E:/SMD_data/SMD_TFRecord/image_SMD_150_Augmented.tfrecords'
data = data_1_5
# 创建 TFRecord 文件
with tf.io.TFRecordWriter(record_file) as writer:
    num_images = len(data)
    batch_size = 50
    processed_images = 0
    for i, (filename, label) in enumerate(data, start=1):
        sample_counter += 1
        img_bgr = cv.imread(filename, 1)
        img_bgr = cv.resize(img_bgr, (150, 150))
        img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

        # 执行多个数据增强操作
        augmented_images = augment_image(img_rgb)

        for augmented_image in augmented_images:
            height = augmented_image.shape[0]
            width = augmented_image.shape[1]
            channel = augmented_image.shape[2]
            image_bytes = augmented_image.tobytes()
            tf_example = image_example(image_bytes, height, width, channel, label)
            writer.write(tf_example.SerializeToString())
        processed_images += 1

        # 输出进度更新
        if processed_images % batch_size == 0:
            if processed_images % (batch_size * 10) == 0:
                print(f'-{processed_images} / {len(train_filenames)}')  # 输出一行连字符
            else:
                print('-', end='')
    # 所有图像处理完成后输出最终进度
    print(f'Processed {processed_images} out of {num_images} images')
    # feature = {
    #     'num_samples': _int64_feature(sample_counter),
    # }

    # tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    # writer.write(tf_example.SerializeToString())
    # print('end!')

# with tf.io.TFRecordWriter(record_file) as writer:
#     feature = {
#         'num_samples': _int64_feature(sample_counter),
#     }

#     tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
#     writer.write(tf_example.SerializeToString())
