# Malaria-Cell-Images-Dataset-CNN

## Project Overview
This project implements a Convolutional Neural Network (CNN) for detecting malaria-infected cells from cell images. The model achieves high accuracy in distinguishing between parasitized and uninfected cells, demonstrating the potential of deep learning in medical diagnosis.

## Dataset
The dataset consists of cell images categorized into two classes:
- Parasitized: infected
- Uninfected: Uninfected

## Model Architecture
- CNN with 2 convolutional layers
- MaxPooling layers for dimension reduction
- Dense layers with dropout for classification
- Input shape: (64, 64, 3)
- Output: Binary classification (Parasitized/Uninfected)

## Key Features
### Data Augmentation
- Rotation range: 20°
- Width/Height shift: 20%
- Horizontal flip
- Zoom range: 20%

### Training Strategy
- Train/Validation/Test split: 64/16/20
- Early stopping with patience=5
- Batch size: 32
- Maximum epochs: 30

## Performance
- Test Accuracy: 96%
- Balanced performance across classes:
  - Parasitized: F1-score = 0.96
  - Uninfected: F1-score = 0.96

## Requirements
```
numpy
matplotlib
scikit-image
opencv-python
tensorflow
scikit-learn
seaborn
```

## Model Evaluation
The model includes comprehensive evaluation metrics:
- ROC curves
- Confusion matrix
- Classification report
- Training/validation curves

Dataset Link On Kaggle : https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria

Code On Kaggle : [https://www.kaggle.com/code/solimankhalil/cnn-model-malaria-dataset](https://www.kaggle.com/code/solimankhalil/cnn-model-malaria-dataset)
