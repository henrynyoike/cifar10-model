# CIFAR-10 Image Classification with PyTorch

A Convolutional Neural Network (CNN) trained on the **CIFAR-10 dataset** to explore image classification using PyTorch and deep-learning techniques.

## Overview

This project implements a CNN that learns to classify images into 10 different object categories using the CIFAR-10 dataset.

The model was trained for **500 epochs** and achieved a reported classification accuracy of **76%**.

## Dataset

CIFAR-10 consists of 60,000 colour images (32×32 pixels) divided into 10 classes:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

The dataset contains 50,000 training images and 10,000 test images.

## Technologies

- Python
- PyTorch
- Torchvision
- NumPy

## Model Performance

| Metric | Result |
|---|---:|
| Architecture | Convolutional Neural Network |
| Dataset | CIFAR-10 |
| Training Epochs | 500 |
| Reported Accuracy | 76% |
| Framework | PyTorch |

## Project Objectives

- Explore convolutional neural networks.
- Understand image classification workflows.
- Apply deep-learning techniques using PyTorch.
- Experiment with model training and evaluation.
- Investigate the challenges of classifying low-resolution images.

## Running the Project

Clone the repository:

```bash
git clone https://github.com/henrynyoike/cifar10-model
cd cifar10-model
```

Install the dependencies:

```bash
pip install torch torchvision numpy
```

Run the training or evaluation script:

```bash
python main.py
```

*Adjust the script name according to the repository structure.*

## License

This project is available for educational and research purposes.