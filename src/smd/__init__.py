"""
文件功能：SMD 表面微放电图像识别项目的 Python 包入口。
代码分布：核心工具位于 dataset.py、config.py 和 shufflenetv2.py。
整理思路：保持包入口轻量，不在导入包时触发 TensorFlow 模型构建或数据读取。
使用方法：在 scripts 中将 src 加入 Python 路径后，使用 from smd.xxx import xxx。
"""

"""Utilities for surface micro-discharge image recognition."""

