# CS4375 Assignment 2 - Neural Networks for Sentiment Analysis

**Full Name:** Purav Patel
**Net ID:** pxp220084
**GitHub Repository:** https://github.com/purav100000/Assignment1_CS4375_Purav_Patel

## Overview

This project implements two neural network architectures for 5-class sentiment analysis on Yelp reviews:
- **Feedforward Neural Network (FFNN)** — Uses bag-of-words input with a single hidden layer
- **Recurrent Neural Network (RNN)** — Processes sequential word embeddings using a vanilla RNN

## Project Structure

```
assignment1/
├── ffnn.py                  # Feedforward Neural Network implementation
├── rnn.py                   # Recurrent Neural Network implementation
├── README.md                # This file


## Environment Setup

### 1. Install Miniconda

Download and install Miniconda for your system from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html). For macOS ARM (M1/M2/M3/M4), download the Apple Silicon `.sh` installer and run:

```bash
bash ~/Downloads/Miniconda3-latest-MacOSX-arm64.sh
```

After installation, close and reopen your terminal.

### 2. Create the Conda Environment

```bash
conda create -n cs4375 python=3.8
```

### 3. Activate the Environment

Before running any commands, you must activate the conda environment:

```bash
conda activate ./cs4375
```

> **Important:** You need to run this activation command every time you open a new terminal session before running any of the Python scripts.

### 4. Install Dependencies

```bash
pip install torch numpy tqdm
```

## Usage

### Running FFNN

```bash
python ffnn.py --hidden_dim 32 --epochs 5 --train_data ./training.json --val_data ./validation.json
```

### Running RNN

```bash
python rnn.py --hidden_dim 64 --epochs 10 --train_data ./training.json --val_data ./validation.json
```

### Command Line Arguments

| Argument | Description |
|---|---|
| `-hd`, `--hidden_dim` | Hidden layer dimension (required) |
| `-e`, `--epochs` | Number of training epochs (required) |
| `--train_data` | Path to training JSON file (required) |
| `--val_data` | Path to validation JSON file (required) |
| `--test_data` | Path to test JSON file (optional)(THE TA - Ruosen Li -  in the teams meeting 3/11/26 said do not have to include) |

## Code Changes

### FFNN — `forward()` Implementation

```python
def forward(self, input_vector):
    hidden = self.activation(self.W1(input_vector))
    output = self.W2(hidden)
    predicted_vector = self.softmax(output)
    return predicted_vector
```

### RNN — `forward()` Implementation

```python
def forward(self, inputs):
    _, hidden = self.rnn(inputs)
    output = self.W(hidden)
    output = torch.sum(output, dim=0)
    predicted_vector = self.softmax(output)
    return predicted_vector
```

## Results Summary

| Model | Hidden Dim | Best Val Acc | Stopped at Epoch |
|---|---|---|---|
| FFNN | 32 | 55.9% | 3 |
| FFNN | 64 | 56.1% | 3 |
| RNN | 16 | 36.1% | 3 (early stop) |
| RNN | 64 | 38.8% | 5 (early stop) |

## Key Differences Between FFNN and RNN

- **Input:** FFNN uses bag-of-words vectors; RNN uses pretrained 50-dim word embeddings
- **Optimizer:** FFNN uses SGD with momentum; RNN uses Adam
- **Stopping:** FFNN runs for a fixed number of epochs; RNN uses early stopping to prevent overfitting
- **Processing:** FFNN processes the entire review as one vector; RNN processes word by word sequentially
