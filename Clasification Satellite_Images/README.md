# 🛰️ Satellite Image Classification with PyTorch  

This project implements a **satellite image classification model** using **PyTorch** and **Convolutional Neural Networks (CNNs)**. We use **ResNet-18** as the base model for feature extraction and supervised training.  

## 📌 Project Features  

**Data Processing:**  
- Automatic extraction and organization of the dataset.  
- Creation of training, validation, and test subsets.  

**Training with Transfer Learning:**  
- **ResNet-18** pretrained on ImageNet.  
- Freezing early layers and fine-tuning the final layer.  

**Model Evaluation:**  
- Calculation of metrics such as **accuracy** and **F1-score**.  
- Visualization of the **confusion matrix**.  

 **Results Visualization:**  
- **Loss and accuracy** plots over epochs.  
- Analysis of the **class distribution** in the dataset.  

## 🔧 Technologies Used  

- **PyTorch** for deep learning model development.  
- **Torchvision** for image transformations.  
- **OpenCV and Matplotlib** for data visualization.  
- **Scikit-learn** for performance analysis.

## 📌 General Workflow  
1. **Importing Required Packages and Modules**
   The following packages were used for various stages of the project:
```python
import os
import random
import shutil
import zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from torchmetrics.classification import Accuracy, F1Score
from sklearn.metrics import confusion_matrix, classification_report

```
