import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import os
from PIL import Image
import random
import numpy as np
from tqdm import tqdm
import time



# Definir transformaciones para preprocesamiento
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
}

# Cargar datasets
train_dataset = torchvision.datasets.ImageFolder(root='data/xrays/train', transform=data_transforms['train'])
test_dataset = torchvision.datasets.ImageFolder(root='data/xrays/test', transform=data_transforms['test'])

# Crear DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Cargar modelo pre-entrenado ResNet50 y modificar la capa final
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid()
)

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Definir función de pérdida y optimizador
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Entrenamiento
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    history = {'accuracy': [], 'loss': []}
    model.train()
    
    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Barra de progreso para la época
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Actualizar barra de progreso con la pérdida promedio actual
            progress_bar.set_postfix(loss=running_loss / (total / labels.size(0)))
        
        epoch_time = time.time() - start_time
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)

        print(f"⏳ Epoch {epoch+1}/{epochs} completada en {epoch_time:.2f} segundos | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")
    
    return history

# Entrenar el modelo


history = train_model(model, train_loader, criterion, optimizer, epochs=10)


# Evaluación del modelo
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device, dtype=torch.float32)
            outputs = model(inputs).squeeze()
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)

labels, predictions = evaluate_model(model, test_loader)

# Calcular métricas
accuracy = accuracy_score(labels, predictions)
f1 = f1_score(labels, predictions)
print(f"Test Accuracy: {accuracy:.3f}")
print(f"Test F1-score: {f1:.3f}")
print(classification_report(labels, predictions, target_names=['Pneumonia', 'Normal']))

# Graficar pérdida y precisión
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Train Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Train Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.show()

# Matriz de confusión
cm = confusion_matrix(labels, predictions)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pneumonia', 'Normal'], yticklabels=['Pneumonia', 'Normal'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
