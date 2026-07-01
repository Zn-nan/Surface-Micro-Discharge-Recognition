# Surface Micro-Discharge Recognition

This repository contains code for visible-light image recognition of surface micro-discharge (SMD) modes using a lightweight ShuffleNetV2 model.

## Paper

**Lightweight Real-Time Recognition of Surface Micro-Discharge Modes Using Visible-Light Discharge Images**

## Dataset

DOI: [10.21227/ecye-se77](https://dx.doi.org/10.21227/ecye-se77)

Download the dataset from IEEE DataPort and place local TFRecord files under:

```text
SMD_data/SMD_TFRecord/
```

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
|   |-- make_tfrecords.py
|   |-- train.py
|   |-- export_tflite.py
|   |-- evaluate.py
|   `-- preview_tfrecord.py
`-- src/
    `-- smd/
        |-- __init__.py
        |-- config.py
        |-- dataset.py
        `-- shufflenetv2.py
```

## Code Layout

- `src/smd/config.py`: common paths, image size, class count, and model filenames.
- `src/smd/dataset.py`: TFRecord parsing and dataset loading utilities.
- `src/smd/shufflenetv2.py`: shared ShuffleNetV2 model definition.
- `scripts/`: runnable experiment scripts.
- `device/`: OpenMV deployment code.
- `models/`: trained Keras and TensorFlow Lite model files.

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

Create TFRecords:

```bash
python scripts/make_tfrecords.py
```

Preview a TFRecord sample:

```bash
python scripts/preview_tfrecord.py
```

Train ShuffleNetV2:

```bash
python scripts/train.py
```

Export TensorFlow Lite:

```bash
python scripts/export_tflite.py
```

Evaluate models:

```bash
python scripts/evaluate.py
```

## Notes

The original experiments were developed in Jupyter notebooks. This repository keeps the notebook logic as runnable Python scripts, while shared dataset and model code is maintained under `src/smd`.
