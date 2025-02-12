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

3. **Data Visualization**
```python
# Contar imágenes por clase
class_counts = {cls: len(os.listdir(os.path.join(output_dir, cls))) for cls in classes}
plt.figure(figsize=(10, 6))
plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
plt.xlabel('Clases')
plt.ylabel('Cantidad de Imágenes')
plt.title('Distribución de Imágenes por Clase')
plt.xticks(rotation=45)
plt.show()
```

4. **Building the Models**

```python
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
```

5. **Training the Models**
```python


# Función para graficar precisión y pérdida
def plot_metrics(losses, accuracies):
    # Graficar precisión y pérdida
    epochs = range(1, len(losses) + 1)
    
    # Crear la figura y los ejes
    fig, ax1 = plt.subplots()

    # Graficar la función de pérdida en el eje izquierdo
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(epochs, losses, color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Crear el segundo eje para la precisión
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='tab:blue')
    ax2.plot(epochs, accuracies, color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # Mostrar el gráfico
    plt.title('Loss and Accuracy over Epochs')

    # Guardar la imagen como archivo JPG
    plt.savefig("metrics_plot.jpg", format='jpg')  # Guardamos la figura como un archivo JPG
    plt.show()


# Función de entrenamiento
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    # Listas para almacenar métricas
    train_losses = []
    train_accuracies = []

    model.train()
    for epoch in range(epochs):
        running_loss, running_corrects = 0.0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += (outputs.argmax(dim=1) == labels).sum().item()

        # Calcular la pérdida y precisión media por época
        epoch_loss = running_loss / len(train_dataset)
        epoch_accuracy = running_corrects / len(train_dataset)

        # Almacenar los valores
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        print(f"Epoch {epoch+1}: Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    # Graficar precisión y pérdida
    plot_metrics(train_losses, train_accuracies)

# Entrenar modelo
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.01)
train_model(model, train_loader, criterion, optimizer, epochs=10)

```

6. **Model Evaluation**
```Python
# Evaluación del modelo
def evaluate_model(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
    return torch.tensor(all_preds), torch.tensor(all_labels)

all_preds, all_labels = evaluate_model(model, test_loader)
test_accuracy = Accuracy(task="multiclass", num_classes=len(classes))(all_preds, all_labels).item()
test_f1 = F1Score(task="multiclass", num_classes=len(classes))(all_preds, all_labels).item()
print(f"Test Accuracy: {test_accuracy:.3f}, Test F1-Score: {test_f1:.3f}")


```

7. **Model Performance Visualization**
```Python

# Matriz de confusión
conf_matrix = confusion_matrix(all_labels.numpy(), all_preds.numpy())
plt.figure(figsize=(8, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=dataset.classes, yticklabels=dataset.classes)
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión")
# Guardar la imagen de la matriz de confusión
plt.savefig("confusion_matrix.jpg", format='jpg')  # Guardamos la figura como un archivo JPG
plt.show()

# Reporte de clasificación
print(classification_report(all_labels.numpy(), all_preds.numpy(), target_names=dataset.classes))

``
