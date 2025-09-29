# Age and Gender Prediction from Face Images

This project implements a **deep learning model** to predict **age** and **gender** from facial images. The model uses a **pretrained ResNet18 backbone** with a custom multi-output head for regression (age) and classification (gender). Advanced **data augmentation** is applied to improve generalization.

---

## Features

* **Age Prediction**: Predicts a continuous age value (normalized).
* **Gender Classification**: Binary gender prediction using raw logits with BCE loss.
* **Pretrained Backbone**: Leverages ResNet18 pretrained on ImageNet for feature extraction.
* **Custom Head**: Intermediate fully-connected layers with ReLU activation and dropout for robust predictions.
* **Data Augmentation**:

  * Random horizontal flips
  * Random rotations
  * Color jitter (brightness, contrast, saturation, hue)
  * Random grayscale
  * CoarseDropout (similar to Cutout)
  * Shift, scale, rotate transforms
* **Training Techniques**:

  * Fine-tuning only top layers (layer3, layer4, and custom head)
  * AdamW optimizer with weight decay
  * Learning rate scheduling using ReduceLROnPlateau
  * Mixed precision training with GradScaler

---

## Usage

1. Install dependencies:

```bash
pip install torch torchvision albumentations opencv-python
```

2. Prepare dataset:

   * Images of faces
   * Labels for age (continuous) and gender (binary)

3. Apply transformations for training and validation using Albumentations.

4. Train the model:

```python
model, age_loss, gender_loss, optimizer, scheduler, scaler = get_model()
# training loop here
```

5. Evaluate on validation set to measure **MAE for age** and **accuracy for gender**.

---

## Results

* Training MAE and accuracy improve consistently.
* Validation MAE/accuracy stabilizes after several epochs, showing strong generalization.
* Strong data augmentation helps reduce overfitting and improves model robustness.

---

## Future Work

* Explore larger backbones (ResNet34/50) for higher accuracy.
* Experiment with **MixUp or CutMix** for further generalization.
* Add **multi-task loss weighting** to balance age and gender predictions.

---

## License

MIT License
