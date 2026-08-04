# Road Anomaly Detection

## Overview

This repository contains the implementation of a deep learning framework for **road anomaly detection** using sequential vehicle telemetry data. The project reproduces and extends the experiments described in the corresponding research paper by training a sequence model to distinguish between normal and anomalous driving behavior.


- Paper: https://iris.polito.it/retrieve/handle/11583/2982400/671449

The implementation provides a complete pipeline including:

* Dataset preprocessing
* Sliding window sequence generation
* Model training
* Model checkpointing
* Inference and anomaly scoring
* Experimental notebooks used during development

The repository is organized to make both research reproduction and future experimentation straightforward.

---

# Repository Structure

```text
Joanna_Research_code/
│
├── checkpoints/
│   ├── best_model.pt          # Best model based on validation metric
│   └── last_model.pt          # Latest checkpoint
│
├── notebook/
│   └── anomaly_detection.ipynb
│       # Research notebook containing all experiments,
│       # visualizations, debugging and evaluation
│
├── src/
│   ├── config.py              # Hyperparameters and configuration
│   ├── dataset_loader.py      # Dataset preprocessing and dataloaders
│   ├── model.py               # Model architecture
│   ├── trainer.py             # Training and evaluation pipeline
│   ├── predict.py             # Inference script
│   ├── utility.py             # Helper utilities
│   └── main.py                # Training entry point
│
└── README.md
```

---

# Features

* Sliding-window sequence generation
* Automatic dataset normalization
* Modular PyTorch implementation
* Checkpoint saving
* Evaluation utilities
* AUROC-based anomaly scoring
* Research notebook containing complete experimentation
* Easily extensible architecture for future research

---

# Requirements

Typical dependencies include

```text
Python >= 3.10

PyTorch
NumPy
scikit-learn
tqdm
matplotlib
```

Install the dependencies using

```bash
pip install -r requirements.txt
```

or install them manually if a requirements file is not provided.

---

# Dataset

The implementation expects the RoAD dataset to be available and properly configured.

The dataset loader performs:

* loading recordings
* normalization
* sliding window generation
* label extraction
* PyTorch dataset creation

Each sequence is converted into fixed-length windows before being passed to the network.

---

# Training

Model training is performed through

```bash
python src/main.py
```

During training the framework

* loads the dataset
* builds the model
* trains for the configured number of epochs
* evaluates performance
* stores checkpoints automatically

Saved checkpoints are written to

```text
checkpoints/
├── best_model.pt
└── last_model.pt
```

where

* **best_model.pt** contains the highest-performing model
* **last_model.pt** contains the latest training state

---

# Inference

Inference is performed using

```bash
python src/predict.py
```

The inference pipeline

1. Loads the trained checkpoint
2. Processes the test dataset
3. Computes anomaly scores
4. Produces evaluation statistics

Typical statistics include

```text
Normal Mean
Anomaly Mean

Normal Std
Anomaly Std

Minimum Score
Maximum Score
```

These statistics can be used to analyze score separation between normal and anomalous sequences.

---

# Experimental Notebook

All experiments performed during the research process are available in

```text
notebook/anomaly_detection.ipynb
```

The notebook contains

* dataset exploration
* preprocessing experiments
* training runs
* inference experiments
* threshold analysis
* evaluation metrics
* debugging
* visualization of anomaly scores

This notebook serves as the complete research log used during implementation.

---

# Project Workflow

```text
Dataset
    │
    ▼
Normalization
    │
    ▼
Sliding Window Generation
    │
    ▼
PyTorch Dataset
    │
    ▼
DataLoader
    │
    ▼
Model Training
    │
    ▼
Checkpoint Saving
    │
    ▼
Inference
    │
    ▼
Anomaly Scores
    │
    ▼
Evaluation Metrics
```

---

# Configuration

Most hyperparameters can be modified inside

```text
src/config.py
```

Typical configurable parameters include

* window size
* stride
* learning rate
* batch size
* optimizer
* number of epochs
* model dimensions
* checkpoint paths

---

# Code Organization

### `dataset_loader.py`

Responsible for

* loading recordings
* preprocessing
* normalization
* sliding-window generation
* PyTorch dataset creation

---

### `model.py`

Contains the neural network architecture used for anomaly detection.

---

### `trainer.py`

Implements

* training loop
* validation
* checkpointing
* loss computation
* metric computation
* prediction utilities

---

### `predict.py`

Responsible for

* loading trained checkpoints
* running inference
* computing anomaly scores
* reporting evaluation statistics

---

### `utility.py`

Provides helper functions used throughout the project.

---

### `main.py`

Primary training entry point.

Responsible for

* model initialization
* optimizer creation
* dataloader setup
* launching training

---

# Outputs

The framework produces

* trained model checkpoints
* anomaly scores
* evaluation statistics
* AUROC measurements
* prediction outputs


---

## Pretrained Weights

Pretrained model weights are included in this repository to facilitate reproducibility and rapid evaluation.

The released checkpoints are located in:

```text
checkpoints/
├── best_model.pt
└── last_model.pt
```

- `best_model.pt` contains the model achieving the best validation performance during training.
- `last_model.pt` corresponds to the final training epoch.

These checkpoints can be directly used with the inference script:

```bash
python src/predict.py
```

---

# Reproducibility

To reproduce the reported experiments

1. Prepare the dataset.
2. Configure hyperparameters in `config.py`.
3. Run

```bash
python src/main.py
```

4. After training completes, the best-performing checkpoint will be stored in

```text
checkpoints/best_model.pt
```

5. Evaluate the model using

```bash
python src/predict.py
```

---

# Extending the Framework

The modular implementation allows researchers to replace or extend individual components independently.

Possible research directions include

* replacing the sequence model with Transformer-based architectures
* experimenting with alternative anomaly scoring methods
* modifying the window generation strategy
* evaluating different normalization techniques
* incorporating multimodal sensor information
* benchmarking on additional autonomous driving datasets

---

# Citation

If this implementation contributes to your research, please cite the original paper that motivated this work together with this repository.

---

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

This repository is provided to support the reproducibility of the accompanying research paper. If you use this implementation in your research, please cite the original paper as well as this repository where appropriate.
