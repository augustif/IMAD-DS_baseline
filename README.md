# IMAD-DS — Industrial Multi-Sensor Anomaly Detection Framework

> A model-agnostic, extensible benchmarking framework for anomaly detection on the [IMAD-DS dataset](https://zenodo.org/doi/10.5281/zenodo.12636236).  
> Built as a community extension of [IMAD-DS_baseline](https://github.com/augustif/IMAD-DS_baseline).

[![Dataset DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.12665499-blue.svg)](https://doi.org/10.5281/zenodo.12665499)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.13%2B-orange.svg)](https://pytorch.org/)

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Download the Dataset](#1-download-the-dataset)
  - [2. Configure Your Experiment](#2-configure-your-experiment)
  - [3. Run Training and Evaluation](#3-run-training-and-evaluation)
  - [4. Interactive Exploration](#4-interactive-exploration)
- [Supported Models](#supported-models)
- [Experiment Tracking](#experiment-tracking)
- [Output Files](#output-files)
- [Contributing a New Model](#contributing-a-new-model)
- [Roadmap](#roadmap)
- [Citation](#citation)
- [License](#license)

---

## Overview

This repository is the main framework for training, evaluating and benchmarking anomaly detection models on the **IMAD-DS** dataset — a multi-rate, multi-sensor industrial dataset designed to test models under real-world **domain shift** conditions.

It extends the original [IMAD-DS_baseline](https://github.com/augustif/IMAD-DS_baseline) — which provided a single Autoencoder (AE) proof-of-concept — into a **fully modular pipeline** where the model is a swappable component. Every other stage (data loading, windowed preprocessing, HDF5 caching, training loop, AUC scoring, domain-shift evaluation, and MLflow tracking) is shared across all models, enabling fair and reproducible comparisons.

Key additions over the baseline:

- **Multi-model support** via a `ModelFactory` registry — add a model by dropping a class into `models/` and updating the config.
- **Convolutional feature extraction** — CNN-based backbones for learning local patterns across sensor channels and time.
- **Hydra configuration** — all hyperparameters, sensor selection, and pipeline steps are managed through structured YAML files, with no code changes needed between runs.
- **MLflow experiment tracking** — every run logs parameters, per-sensor reconstruction losses, and AUC scores automatically.
- **Domain-shift-aware evaluation** — AUC is computed globally, and split by source vs. target domain, following the protocol defined in the original paper.
- **Scriptable entry point** (`main.py`) alongside an interactive notebook (`deploy.ipynb`) for exploratory use.

---

## Dataset

**IMAD-DS** (Industrial Multi-sensor Anomaly Detection Dataset) is available on Zenodo:

> 📦 [https://zenodo.org/doi/10.5281/zenodo.12636236](https://zenodo.org/doi/10.5281/zenodo.12636236)

The dataset covers two scaled industrial machines recorded with three sensors simultaneously:

| Machine | Anomaly Type |
|---|---|
| **Robotic Arm** | Bolts removed at arm nodes → mechanical imbalance |
| **Brushless Motor** | Magnet displacement (oscillations) · Belt tightening (mechanical stress) |

| Sensor | Sampling Rate |
|---|---|
| Analog Microphone (`imp23absu_mic`) | 16 kHz |
| 3-axis Accelerometer (`ism330dhcx_acc`) | 6.7 kHz |
| 3-axis Gyroscope (`ism330dhcx_gyro`) | 6.7 kHz |

Each machine's recordings are split into a **source domain** (abundant normal training data) and a **target domain** (limited training data), reflecting the industrial reality that models trained under controlled conditions must generalise to noisier, different-load deployment environments.

Data is stored in `.parquet` files, with `.csv` metadata files providing per-segment operational and environmental condition labels.

---

## Repository Structure

```
IMAD-DS/
│
├── main.py                  # CLI entry point (Hydra + MLflow)
├── utilities.py             # Device selection, seeding, helpers
├── deploy.ipynb             # Interactive notebook for exploration
├── requirements.txt         # Full dependency list
│
├── conf/                    # Hydra configuration files
│   └── config.yaml          # Root config (dataset, model, pipeline, paths)
│
├── datasets/                # PyTorch Dataset classes
│   └── dataset_IMADS.py     # IMADSDatasetTrain, IMADSDatasetTest, IMADSBaseDataset
│
├── preprocessing/           # Preprocessing pipeline
│   └── prepr_pipelines.py   # PreprocessingPipeline (windowing, transforms)
│
├── models/                  # Model registry and training manager
│   ├── __init__.py          # ModelFactory + IMADSModelManager
│   └── ...                  # Individual model implementations
│
└── metrics/                 # Evaluation utilities
    └── utils.py             # AUC computation, segment-level aggregation, domain-shift metrics
```

---

## Requirements

- Python 3.7 or higher
- PyTorch 1.13+ (install separately — see below)
- All other dependencies listed in `requirements.txt`

Core libraries include: `hydra-core`, `omegaconf`, `mlflow`, `numpy`, `pandas`, `scikit-learn`, `librosa`, `h5py`, `pyarrow`, `fastparquet`, `matplotlib`, `seaborn`.

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/augustif/IMAD-DS.git
cd IMAD-DS
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Install PyTorch**

Install the version of PyTorch that matches your hardware (CPU, CUDA, or Apple Silicon MPS). See [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for the correct command. The framework detects the available device automatically (`cuda` → `mps` → `cpu`).

Example for CUDA 11.8:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Usage

### 1. Download the Dataset

Download and extract the IMAD-DS dataset from Zenodo:

> [https://zenodo.org/doi/10.5281/zenodo.12636236](https://zenodo.org/doi/10.5281/zenodo.12636236)

Download `BrushlessMotor.7z` and/or `RoboticArm.7z` and extract them into your data folder. Then update `conf/config.yaml` to point `dataset.data_folder` to that location.

### 2. Configure Your Experiment

All configuration is managed through `conf/config.yaml` (and any Hydra overrides). The key fields are:

```yaml
experiment_name: my_experiment
run_name: ae_run_01

dataset:
  data_folder: /path/to/imad-ds/data
  dataset_name: IMAD-DS
  machine: BrushlessMotor          # or RoboticArm
  window_size_ms: 100
  preload_and_transform: true
  batch_process: false
  load_preprocessed: false         # set true to reuse cached HDF5

sensors_enabled:
  f_ism330dhcx_acc: true
  s_ism330dhcx_acc: true
  f_ism330dhcx_gyro: true
  s_ism330dhcx_gyro: true
  f_imp23absu_mic: true
  s_imp23absu_mic: true

model:
  model_type: AE                   # e.g. AE, CNN_AE — set by ModelFactory
  seed: 42
  batch_size: 64
  num_epochs: 50
  lr: 0.001
  valid_size: 0.1
  retrain: false
  save_after_n_epochs: 10

preprocess_pipeline:
  order: [windowing]               # list of preprocessing steps to apply

test:
  use_best_model: true
  group_by_segment_id: true
  aggregation_type: mean
  domain_shift: true               # compute source vs. target AUC split

paths:
  checkpoint_path: checkpoints/
  checkpoint_name: model.pt

ml_tracking:
  path: mlruns/
  model_path: model
  model_name: IMADS_model
```

Any value can be overridden from the command line using standard Hydra syntax.

### 3. Run Training and Evaluation

```bash
python main.py
```

To override config values without editing the YAML file:

```bash
# Change machine
python main.py dataset.machine=RoboticArm

# Change model type
python main.py model.model_type=CNN_AE

# Increase epochs and batch size
python main.py model.num_epochs=100 model.batch_size=128

# Disable microphone, run on CPU only
python main.py sensors_enabled.f_imp23absu_mic=false sensors_enabled.s_imp23absu_mic=false
```

Hydra also supports multi-run sweeps:

```bash
python main.py --multirun dataset.machine=BrushlessMotor,RoboticArm model.model_type=AE,CNN_AE
```

### 4. Interactive Exploration

For step-by-step experimentation, open the notebook:

```bash
jupyter notebook deploy.ipynb
```

This notebook covers the same pipeline as `main.py` in an interactive format, suitable for visualising intermediate outputs, debugging preprocessing steps, and inspecting model internals.

---

## Supported Models

Models are registered through `ModelFactory` in `models/__init__.py`. The factory is keyed by `model_type` in the config.

| `model_type` | Description |
|---|---|
| `AE` | Fully-connected Autoencoder. Reconstruction error is the anomaly score. Reference model from the original baseline. |
| `CNN_AE` | Convolutional Autoencoder. A CNN encoder extracts local temporal patterns across sensor channels before the bottleneck. Well-suited to the audio and vibration signals in IMAD-DS. |

Both models are trained on normal data only. At test time, high reconstruction loss on a window indicates anomalous behaviour. Per-sensor losses are computed separately, in addition to an overall loss, allowing analysis of which sensor is most informative for each machine.

---

## Experiment Tracking

Every run is tracked with **MLflow**. The tracking UI can be launched with:

```bash
mlflow ui --backend-store-uri mlruns/
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

Each run logs:

- All Hydra config parameters
- Per-sensor mean and std of reconstruction loss (train and test)
- AUC scores per sensor and overall
- Domain-shift AUC breakdown (source / target / combined)
- All output CSV files as artifacts

---

## Output Files

After a run, the checkpoint folder (`checkpoints/<experiment_name>/<machine><run_name>/`) contains:

| File | Contents |
|---|---|
| `model.pt` | Best model checkpoint (lowest validation loss) |
| `reconstruction_scores.csv` | Per-window reconstruction losses and ground-truth labels |
| `reconstruction_scores_by_segment_id.csv` | Losses aggregated to recording-segment level |
| `AUC_scores.csv` | AUC-ROC per sensor and overall |
| `AUC_scores_by_segment_id.csv` | AUC-ROC at segment level |
| `AUC_scores_domain_shift.csv` | AUC split by source / target / combined domain |
| `AUC_scores_by_segment_id_domain_shift.csv` | Same, at segment level |

Results are also published to MLflow automatically.

---

## Contributing a New Model

The framework is designed so that adding a new model requires minimal changes outside the `models/` directory.

**Step 1 — Implement your model class**

Create `models/my_model.py`. Your class must expose at minimum:

- `__init__(self, cfg_model: dict)` — receives the merged config dict (includes `window_lengths`, `num_channels`, `sensors_enabled` in addition to all fields under `model` in the YAML).
- `forward(self, x)` — standard PyTorch forward pass.
- A reconstruction-based loss compatible with `IMADSModelManager`.

**Step 2 — Register it in the factory**

In `models/__init__.py`, add your model to the `ModelFactory` lookup:

```python
from models.my_model import MyModel

_MODEL_REGISTRY = {
    "AE": AutoEncoder,
    "CNN_AE": ConvAutoEncoder,
    "MY_MODEL": MyModel,          # add this line
}
```

**Step 3 — Add a config**

In `conf/config.yaml` (or a separate Hydra config group), set `model.model_type: MY_MODEL` and add any model-specific hyperparameters under the `model` key.

**Step 4 — Run and share results**

Run your model on both machines (`BrushlessMotor`, `RoboticArm`) across both source and target domains, commit the output CSVs alongside your model file, and open a pull request. Results are compared on AUC-ROC, reported in the format used in the original paper.

---

## Roadmap

- [ ] **TranAD** — transformer-based anomaly detection with two-phase adversarial training, adapted to the multi-rate sensor structure of IMAD-DS.
- [ ] Broader family of **transformer-based models** — including patch-based approaches treating sensor-channel windows as token sequences, and models with built-in domain adaptation for the source-to-target shift.
- [ ] Consolidated **results leaderboard** in the repository wiki.
- [ ] Automated CI run for new model contributions.

---

## Citation

If you use this framework or the IMAD-DS dataset in your work, please cite:

```bibtex
@dataset{augusti_2024_imadds,
  author       = {Augusti, Filippo and Albertini, Davide and Esmer, Kudret and Sannino, Roberto and Bernardini, Alberto},
  title        = {{IMAD-DS: A Dataset for Industrial Multi-Sensor Anomaly Detection Under Domain Shift Conditions}},
  year         = 2024,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.12665499},
  url          = {https://zenodo.org/doi/10.5281/zenodo.12636236}
}
```

The original baseline code:

> [https://github.com/augustif/IMAD-DS_baseline](https://github.com/augustif/IMAD-DS_baseline)

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

The IMAD-DS dataset is licensed under **Creative Commons Attribution Share Alike 4.0 International (CC BY-SA 4.0)**.