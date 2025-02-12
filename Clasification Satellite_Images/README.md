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
2. **Data Loading**
```python

# Definir rutas de los datos
data_dir = 'data/data'
output_dir = 'data/small_subset'

# Eliminar y crear carpetas de salida
def prepare_directories():
    for directory in [output_dir, data_dir]:
        if os.path.exists(directory):
            shutil.rmtree(directory)
    os.makedirs(output_dir, exist_ok=True)
prepare_directories()

# Descomprimir los datos si no existen
def unzip_data():
    if not os.path.exists(data_dir):
        with zipfile.ZipFile('data/data.zip', 'r') as zip_ref:
            zip_ref.extractall('data/')
        print("Datos descomprimidos.")
unzip_data()

# Crear subconjunto de datos
classes = os.listdir(data_dir)
for class_name in classes:
    class_dir = os.path.join(data_dir, class_name)
    output_class_dir = os.path.join(output_dir, class_name)
    os.makedirs(output_class_dir, exist_ok=True)
    images = os.listdir(class_dir)
    sample_size = int(1.0 * len(images))
    sampled_images = random.sample(images, sample_size)
    for image in sampled_images:
        shutil.copy(os.path.join(class_dir, image), os.path.join(output_class_dir, image))
print("Subconjunto de datos creado.")

# Definir transformaciones para los datos
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Cargar dataset y dividir en subconjuntos
dataset = datasets.ImageFolder(root=output_dir, transform=transform)
train_size, val_size = int(0.7 * len(dataset)), int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, shuffle=False)
test_loader = DataLoader(test_dataset, shuffle=False)


```
