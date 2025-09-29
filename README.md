Age and Gender Prediction from Face Images

This project implements a deep learning model to predict age and gender from facial images. The model uses a pretrained ResNet18 backbone with a custom multi-output head for regression (age) and classification (gender). Advanced data augmentation is applied to improve generalization.

Features

Age Prediction: Predicts a continuous age value (normalized).

Gender Classification: Binary gender prediction using raw logits with BCE loss.

Pretrained Backbone: Leverages ResNet18 pretrained on ImageNet for feature extraction.

Custom Head: Intermediate fully-connected layers with ReLU activation and dropout for robust predictions.

Data Augmentation:

Random horizontal flips

Random rotations

Color jitter (brightness, contrast, saturation, hue)

Random grayscale

CoarseDropout (similar to Cutout)

Shift, scale, rotate transforms

Training Techniques:

Fine-tuning only top layers (layer3, layer4, and custom head)

AdamW optimizer with weight decay

Learning rate scheduling using ReduceLROnPlateau

Mixed precision training with GradScaler

Usage

Install dependencies:
