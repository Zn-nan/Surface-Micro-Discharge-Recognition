# Surface Micro-Discharge Recognition

Source code for surface micro-discharge (SMD) mode recognition using visible-light discharge images and lightweight deep learning models.

## Paper

**Lightweight Real-Time Recognition of Surface Micro-Discharge Modes Using Visible-Light Discharge Images**

## Dataset

The dataset is publicly available on IEEE DataPort:

https://doi.org/10.21227/ecye-se77

## Structure

```text
.
|-- README.md
|-- requirements.txt
|-- datasets/
|   `-- DataPort_DOI.txt
|-- device/
|   |-- openmv_image_classification.py
|   `-- openmv_my_LCD.py
|-- models/
|   |-- ShuffleNetV2_model-litemodel20260624-9936-9853.tflite
|   `-- ShuffleNetV2_model_checkpoint_SMD_150_Augmented_20260624-9936.h5
|-- scripts/
|   |-- create_tfrecord_augmented.py
|   |-- evaluate_roc_curve.py
|   |-- quantize_shufflenetv2.py
|   |-- read_tfrecord_demo.py
|   `-- train_shufflenetv2.py
`-- src/
    `-- smd_recognition/
        |-- __init__.py
        |-- dataset.py
        `-- shufflenetv2.py
```

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

Create TFRecords:

```bash
python scripts/create_tfrecord_augmented.py
```

Train ShuffleNetV2:

```bash
python scripts/train_shufflenetv2.py
```

Quantize and export TensorFlow Lite:

```bash
python scripts/quantize_shufflenetv2.py
```

Evaluate ROC curve:

```bash
python scripts/evaluate_roc_curve.py
```

## Notes

Download the dataset from IEEE DataPort and place local data according to `datasets/DataPort_DOI.txt`.
The scripts were converted from the original Jupyter notebooks; edit local paths before running on a new machine.
