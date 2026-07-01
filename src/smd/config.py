"""
文件功能：集中保存项目中常用的路径、图像尺寸和类别数量。
代码分布：数据路径放在 DATA_DIR/TFRECORD_DIR；模型输出路径放在 MODEL_DIR；训练常量放在 IMAGE_SIZE、NUM_CLASSES、BATCH_SIZE。
整理思路：把多个脚本反复出现的路径和常量集中到一处，后续迁移电脑或调整实验参数时少改文件。
使用方法：from smd.config import IMAGE_SIZE, TRAIN_TFRECORD，然后在训练、评估或量化脚本中复用。
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 本地数据目录不提交到 GitHub，只记录相对位置。
DATA_DIR = PROJECT_ROOT / "SMD_data"
TFRECORD_DIR = DATA_DIR / "SMD_TFRecord"
MODEL_DIR = PROJECT_ROOT / "models"

IMAGE_SIZE = (150, 150)
NUM_CLASSES = 2
BATCH_SIZE = 128

TRAIN_TFRECORD = TFRECORD_DIR / "image_SMD_150_train_20241209_Augmented.tfrecords"
TEST_TFRECORD = TFRECORD_DIR / "image_SMD_150_test_20241209.tfrecords"

SHUFFLENETV2_H5 = MODEL_DIR / "ShuffleNetV2_model_checkpoint_SMD_150_Augmented_20260624-9936.h5"
SHUFFLENETV2_TFLITE = MODEL_DIR / "ShuffleNetV2_model-litemodel20260624-9936-9853.tflite"
