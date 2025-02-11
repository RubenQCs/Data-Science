import os
import shutil
import zipfile
import random
import glob
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Configurar rutas
data_path = "data/xrays"
zip_path = "data/xrays.zip"
plot_dir = "plots"

# Crear carpeta de gráficos si no existe
os.makedirs(plot_dir, exist_ok=True)

# Eliminar carpeta de datos si existe
if os.path.exists(data_path):
    shutil.rmtree(data_path)
print(f"Carpeta '{data_path}' eliminada correctamente.")

# Descomprimir datos si no existen
if not os.path.exists(data_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_path)
    print("Datos descomprimidos correctamente.")

# Función para guardar gráficos
def save_plot(filename):
    path = os.path.join(plot_dir, filename)
    plt.savefig(path, bbox_inches='tight')
    print(f"Gráfico guardado en: {path}")
    plt.close()

# Preprocesamiento de imágenes
def preprocess_input_with_norm(image):
    return tf.image.per_image_standardization(image)

# Data augmentation y carga de datos
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_with_norm)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_with_norm)

train_data = train_datagen.flow_from_directory(f"{data_path}/train", target_size=(224, 224), batch_size=32, class_mode='binary')
test_data = test_datagen.flow_from_directory(f"{data_path}/test", target_size=(224, 224), batch_size=32, class_mode='binary', shuffle=False)

# Visualizar distribución de datos
class_names = list(train_data.class_indices.keys())
class_counts = [len(os.listdir(os.path.join(f"{data_path}/train", cls))) for cls in class_names]

plt.figure(figsize=(8, 5))
plt.bar(class_names, class_counts, color='skyblue')
plt.xlabel('Clases')
plt.ylabel('Cantidad de imágenes')
plt.title('Cantidad de imágenes por clase en el conjunto de entrenamiento')
plt.xticks(rotation=45)
save_plot("data_distribution.png")

# Aumentación de datos
num_to_generate = abs(class_counts[0] - class_counts[1])
normal_path = f"{data_path}/train/NORMAL"
os.makedirs(normal_path, exist_ok=True)

normal_datagen = ImageDataGenerator(rescale=1./255, rotation_range=15, zoom_range=0.2, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
image_files = glob.glob(os.path.join(normal_path, "*.jpeg"))

for i in range(num_to_generate):
    img_path = random.choice(image_files)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    augmented_img = next(normal_datagen.flow(img, batch_size=1))[0]
    new_filename = os.path.join(normal_path, f"augmented_{i}.jpeg")
    cv2.imwrite(new_filename, (augmented_img * 255).astype(np.uint8))
print(f"Aumento de datos completado: {num_to_generate} imágenes nuevas generadas.")

# Recargar datos después de la aumentación
train_data = train_datagen.flow_from_directory(f"{data_path}/train", target_size=(224, 224), batch_size=32, class_mode='binary')

test_data = test_datagen.flow_from_directory(f"{data_path}/test", target_size=(224, 224), batch_size=32, class_mode='binary', shuffle=False)

# Definir modelo basado en ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
out = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=out)
model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01), loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar modelo
history = model.fit(train_data, epochs=10, validation_data=test_data, shuffle=True)

# Evaluar modelo
predictions = model.predict(test_data)
predictions = np.round(predictions).flatten()
labels = test_data.classes

test_accuracy = accuracy_score(labels, predictions)
test_f1_score = f1_score(labels, predictions)
print(f"\nTest accuracy: {test_accuracy:.3f}\nTest F1-score: {test_f1_score:.3f}")

# Graficar métricas de entrenamiento
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5)
save_plot("accuracy_plot.png")

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5)
save_plot("loss_plot.png")

# Matriz de confusión
cm = confusion_matrix(labels, predictions)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
save_plot("confusion_matrix.png")

# Reporte de clasificación
print(classification_report(labels, predictions, target_names=class_names))
