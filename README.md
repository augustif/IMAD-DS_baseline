# Reproducing Baseline Results on IMAD-DS Dataset

This repository contains scripts to reproduce the baseline results on the IMAD-DS dataset, which is available on Zenodo at [https://zenodo.org/doi/10.5281/zenodo.12636236]. The scripts include Jupyter notebooks and Python scripts for downloading, preprocessing, and training the baseline model.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Download the Dataset](#1-download-the-dataset)
  - [2. Preprocess the Dataset](#2-preprocess-the-dataset)
  - [3. Train the Baseline Model](#3-train-the-baseline-model)
- [Results](#results)
- [License](#license)

## Overview

The repository includes the following scripts:

- `download_dataset.ipynb`: Downloads the IMAD-DS dataset from Zenodo;
- `preprocess.ipynb`: Preprocesses the dataset and stores it in HDF5 format;
- `baseline.ipynb`: Trains the baseline model using the preprocessed dataset.

## Requirements

- Python 3.7 or higher
- Jupyter Notebook
- Required Python packages (listed in `requirements.txt`)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/augustif/IMAD-DS_baseline.git
    cd IMAD-DS_baseline
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
3. Install your preferred version of torch, depending on your CPU/GPU setup

## Usage

### 1. Download the Dataset

Open and run the [`download_dataset.ipynb`](download_dataset.ipynb) notebook to download the IMAD-DS dataset from Zenodo. This notebook will save the dataset to a specified directory and unzip the files.

### 2. Preprocess the Dataset

Open and run the [`preprocess.ipynb`](preprocess.ipynb) notebook to preprocess the dataset. This notebook will:
- Load the raw dataset and the metadata related to it;
- divide the dataset into train and test set based on the information contained in the metadata,
- Perform necessary preprocessing steps, namely windowing;
- Store the preprocessed data in HDF5 format.

### 3. Train the Baseline Model

Open and run the [`baseline.ipynb`](baseline.ipynb) notebook to train the baseline model using the preprocessed dataset. This notebook will:
- Load the preprocessed data;
- Train the model;
- Show the results;
- save the model and the results.

## Results

After running the [`baseline.ipynb`](baseline.ipynb) notebook, the results of the baseline model training will be saved in the specified output directory. You can analyze these results to compare with the baseline performance reported in the original study.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

