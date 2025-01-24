# AVPR Task 2: Action Recognition with Custom CNN and ResNet

This project focuses on classifying images of humans performing various actions (e.g., climbing, drinking, playing, etc.) using deep learning models. The primary goal is to create a custom Convolutional Neural Network (CNN) and compare its performance against a pre-trained ResNet model. The models are evaluated based on their accuracy in recognizing the actions depicted in the images.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Models](#models)
- [How to Use](#how-to-use)
- [Results](#results)
- [References](#references)

## Overview

The project explores the following key objectives:
1. **Custom CNN Implementation**: Design and train a custom CNN from scratch for action classification.
2. **ResNet Comparison**: Use a pre-trained ResNet model (e.g., ResNet-50) as a baseline for comparison.
3. **Grid Search for Hyperparameter Optimization**: Experiment with hyperparameters such as the number of layers, filter sizes, learning rates, and epochs to find the optimal configuration for the custom CNN.

The implementation leverages PyTorch and other popular libraries such as NumPy and Matplotlib for training, evaluation, and visualization.

## Dataset

The dataset used is the **Stanford40 Actions Dataset**, which contains images of humans performing 40 distinct actions, such as:
- Climbing
- Drinking
- Playing an instrument
- Riding a horse
- And more...

The dataset is organized into training and testing splits, and preprocessing steps ensure compatibility with the implemented models.

## Models

### Custom CNN
The `CustomCNN` model is a parameterized Convolutional Neural Network that allows flexibility in:
- Number of layers
- Base filter sizes
- Kernel sizes
- Pooling sizes
- Dropout probabilities

This adaptability enables hyperparameter optimization and experimentation with different architectures.

### Pre-trained ResNet
The project also uses a pre-trained ResNet model (e.g., ResNet-50) from the PyTorch model zoo. ResNet is known for its residual learning mechanism, which enables very deep networks by addressing the vanishing gradient problem.

## Training and Testing

Two Jupyter notebooks are provided for training and evaluation:

- **`test_custom.ipynb`**: Train and evaluate the custom CNN model.
- **`test_resnet.ipynb`**: Fine-tune and evaluate the ResNet model.

Run the notebooks to reproduce the results or adapt them for your experiments.

---

## Data Preprocessing

To prepare the dataset:

1. Ensure the Stanford40 dataset is organized correctly.
2. Use the provided preprocessing script to structure the dataset for training and testing.

---

## Grid Search

The project implements a grid search to optimize hyperparameters for the custom CNN. Parameters such as the number of layers, learning rates, and kernel sizes are tested to achieve the best validation accuracy.

---

## Results

- The custom CNN achieved an accuracy of **X%** on the test set after grid search optimization.
- The pre-trained ResNet model achieved an accuracy of **Y%**, serving as a baseline for comparison.
- ResNet outperformed the custom CNN due to its pre-trained feature extraction capabilities, but the custom CNN demonstrated competitive performance with fewer computational resources.

---

## References

1. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. ["Deep Residual Learning for Image Recognition"](https://arxiv.org/abs/1512.03385). arXiv preprint 2015.
2. [Stanford40 Actions Dataset](http://vision.stanford.edu/Datasets/40actions.html)

