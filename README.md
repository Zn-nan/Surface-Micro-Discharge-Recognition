# Surface Micro-Discharge Recognition

Code for visible-light image recognition of atmospheric-pressure surface micro-discharge (SMD) modes using a lightweight ShuffleNetV2 model.

## Paper

**Lightweight Real-Time Recognition of Surface Micro-Discharge Modes Using Visible-Light Discharge Images**

## Dataset

DOI: [10.21227/ecye-se77](https://dx.doi.org/10.21227/ecye-se77)

The dataset contains visible-light images of SMD plasmas collected under different operating conditions, including applied voltages, dielectric materials, dielectric thicknesses, and camera exposure times.

Labels are determined by synchronized Fourier Transform Infrared (FTIR) measurements of gaseous products, rather than manual visual inspection. This provides objective ground-truth labels for supervised image classification.

| Class | Images |
|---|---:|
| Ozone | 3,926 |
| Non-Ozone | 3,926 |
| Total | 7,852 |

### Label Definition

- `Ozone`: ozone (O3) is the dominant gaseous product, and nitrogen oxides (NO or NO2) are absent.
- `Non-Ozone`: NOx species are detected during the initial discharge stage, including transition mode and NOx mode.

Download the dataset from IEEE DataPort and place local TFRecord files under:

```text
SMD_data/SMD_TFRecord/
```

## Experimental Setup

The SMD device consists of a powered copper electrode, a dielectric barrier, and a grounded steel mesh electrode. A high-voltage AC power supply operating at 8 kHz was used to generate the discharge.

Visible-light discharge images were captured in a darkroom using a Nikon D750 digital camera. Gaseous products were measured using a Bruker VERTEX 70 FTIR spectrometer to determine the corresponding discharge mode labels.

## Repository Structure

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
- `src/smd/dataset.py`: shared TFRecord parsing and dataset loading utilities.
- `src/smd/shufflenetv2.py`: shared ShuffleNetV2 model definition.
- `scripts/`: runnable data preparation, training, export, and evaluation scripts.
- `device/`: OpenMV deployment code for edge inference.
- `models/`: trained Keras and TensorFlow Lite model files.

## Requirements

```bash
pip install -r requirements.txt
```

Main dependencies:

```text
tensorflow==2.6.0
numpy
opencv-python
matplotlib
scikit-learn
pillow
```

## Quick Start

Clone this repository and install the dependencies:

```bash
git clone https://github.com/Zn-nan/Surface-Micro-Discharge-Recognition.git
cd Surface-Micro-Discharge-Recognition
pip install -r requirements.txt
```

Download the dataset from IEEE DataPort and place the TFRecord files under:

```text
SMD_data/SMD_TFRecord/
```

Before running the scripts, check local paths in `src/smd/config.py` and, if needed, update the path variables inside `scripts/*.py`.

Run the workflow:

```bash
python scripts/preview_tfrecord.py
python scripts/train.py
python scripts/export_tflite.py
python scripts/evaluate.py
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

## Applications

This dataset and code can support research on SMD mode recognition, atmospheric-pressure plasma diagnostics, image classification, lightweight neural networks, edge AI, and intelligent plasma monitoring.

## Notes

The original experiments were developed in Jupyter notebooks. This repository keeps the notebook logic as runnable Python scripts, while shared dataset and model code is maintained under `src/smd`.
