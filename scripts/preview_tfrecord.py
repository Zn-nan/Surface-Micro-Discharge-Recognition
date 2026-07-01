"""
文件功能：快速读取一个 TFRecord 文件，并显示其中一张样本图片和标签。
代码分布：从 src/smd/dataset.py 复用 TFRecord 解析逻辑，然后随机取一张图像显示。
整理思路：这是数据检查脚本，不参与训练流程；用于确认 TFRecord 是否写入正确。
使用方法：修改 tfrecord_path 为本地文件路径，然后运行 python scripts/preview_tfrecord.py。
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from smd.dataset import read_tfrecord


tfrecord_path = "SMD_data/SMD_TFRecord/image_SMD_150_train_20241209.tfrecords"

# 不做 one-hot 和归一化，便于直接查看原始图像像素和整数标签。
dataset = read_tfrecord(tfrecord_path, one_hot=False, normalize=False).shuffle(1000)

for image, label in dataset.take(1):
    plt.figure(figsize=(5, 5))
    plt.imshow(image.numpy())
    plt.title(f"Label = {label.numpy()}")
    plt.axis("off")
    plt.show()
