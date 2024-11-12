# Handwritten Digit Recognition Using Neural Networks

This project implements a neural network model to classify handwritten digits from the famous MNIST dataset. The goal is to train a neural network to recognize handwritten digits from 0 to 9 with high accuracy.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Overview

Handwritten digit recognition is a fundamental problem in computer vision and pattern recognition. This project builds a neural network to classify grayscale images of handwritten digits (28x28 pixels) from the MNIST dataset. The model learns to recognize the patterns and features of each digit, making it capable of predicting the digit shown in unseen images with high accuracy.

## Dataset

The project uses the **MNIST dataset**, which contains:
- 60,000 training images
- 10,000 test images

Each image is a 28x28 grayscale image of a single handwritten digit (0-9). The dataset is available in popular machine learning libraries such as `tensorflow` and `torchvision`.

## Model Architecture

The neural network model consists of:
1. **Input Layer** - Takes a 28x28 pixel input, flattened to a vector of size 784.
2. **Hidden Layers** - Two hidden layers with ReLU activations:
   - First hidden layer: 128 neurons
   - Second hidden layer: 64 neurons
3. **Output Layer** - 10 neurons with softmax activation to predict the probability of each digit (0-9).

### Hyperparameters

- **Learning Rate**: 0.001
- **Batch Size**: 32
- **Epochs**: 10

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/username/handwritten-digit-recognition.git
   cd handwritten-digit-recognition
