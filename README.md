# SMD Visible-Light Image Recognition

This repository provides the source code for surface micro-discharge (SMD) mode recognition using visible-light discharge images and lightweight deep learning models.

## Paper

**Lightweight Real-Time Recognition of Surface Micro-Discharge Modes Using Visible-Light Discharge Images**

## Dataset

The dataset is publicly available on IEEE DataPort:

https://doi.org/10.21227/ecye-se77

## Repository Structure

```text
SMD-Visible-Image-Recognition/
│
├── README.md
├── LICENSE
├── requirements.txt
├── train.py
├── test.py
├── export_tflite.py
├── predict.py
│
├── models/
│   ├── ShuffleNetV2.py
│   ├── layers.py
│   └── __init__.py
│
├── utils/
│   ├── dataloader.py
│   ├── metrics.py
│   ├── losses.py
│   └── visualization.py
│
├── deployment/
│   ├── tflite/
│   └── microcontroller/
│
├── examples/
│   ├── demo.jpg
│   ├── result.jpg
│   └── inference_demo.py
│
├── docs/
│   ├── figures/
│   └── model_architecture.png
│
└── datasets/
    └── DataPort_DOI.txt
```

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train.py
```

### Testing

```bash
python test.py
```

### TensorFlow Lite Export

```bash
python export_tflite.py
```

### Inference

```bash
python predict.py
```

## Notes

Please place the dataset according to the instructions in `datasets/DataPort_DOI.txt`.
