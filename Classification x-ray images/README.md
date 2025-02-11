# 🧑‍🔬 Machine Learning & Deep Learning Project  

  This project was developed using **TensorFlow** and **PyTorch** to implement and compare different machine learning and deep learning models. In this repository, we will specifically describe the code implemented using **TensorFlow**. The general steps followed in this project are outlined below.  

  The general steps followed in this project are outlined below.  

## 📌 General Workflow  

1. **Importing Required Packages and Modules**
   The following packages were used for various stages of the project:
   ```python
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

3. **Data Loading**
To load and preprocess the data, the following steps were performed:
```python
# Configure paths
data_path = "data/xrays"
zip_path = "data/xrays.zip"
plot_dir = "plots"

# Create plot directory if it doesn't exist
os.makedirs(plot_dir, exist_ok=True)

# Remove existing data folder if it exists
if os.path.exists(data_path):
    shutil.rmtree(data_path)
print(f"Folder '{data_path}' removed successfully.")

# Unzip data if it doesn't exist
if not os.path.exists(data_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_path)
    print("Data unzipped successfully.")

# Function to save plots
def save_plot(filename):
    path = os.path.join(plot_dir, filename)
    plt.savefig(path, bbox_inches='tight')
    print(f"Plot saved at: {path}")
    plt.close()

# Image preprocessing
def preprocess_input_with_norm(image):
    return tf.image.per_image_standardization(image)

#data loading
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_with_norm)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_with_norm)

train_data = train_datagen.flow_from_directory(f"{data_path}/train", target_size=(224, 224), batch_size=32, class_mode='binary')
test_data = test_datagen.flow_from_directory(f"{data_path}/test", target_size=(224, 224), batch_size=32, class_mode='binary', shuffle=False)

```

3. **Data Visualization**
In this step, we visualize the distribution of the dataset to understand the class balance in the training data.

```python
# Visualize data distribution
class_names = list(train_data.class_indices.keys())
class_counts = [len(os.listdir(os.path.join(f"{data_path}/train", cls))) for cls in class_names]

plt.figure(figsize=(8, 5))
plt.bar(class_names, class_counts, color='skyblue')
plt.xlabel('Classes')
plt.ylabel('Number of Images')
plt.title('Number of Images per Class in the Training Set')
plt.xticks(rotation=45)
save_plot("data_distribution.png")
```
![Data Distribution Plot](plots/data_distribution.png)


4. **Data Preprocessing and Augmentation**

In this step, we preprocess the data and apply data augmentation techniques. Given the imbalance in the dataset, where the "normal" category has fewer images compared to the "pneumonia" category, it is necessary to increase the number of images in the "normal" category. This is addressed by generating new images from the existing ones using augmentation techniques.

```
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

# Load Data
train_data = train_datagen.flow_from_directory(f"{data_path}/train", target_size=(224, 224), batch_size=32, class_mode='binary')

test_data = test_datagen.flow_from_directory(f"{data_path}/test", target_size=(224, 224), batch_size=32, class_mode='binary', shuffle=False)
```
5. **Building the Models**
In this step, we build the deep learning model using *ResNet50** model. We add extra layers on top of the base model, including **Dropout** to reduce overfitting.  Given that this is a binary classification problem, we use a **sigmoid** activation function at the output layer

```python
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
```
  
7. **Training the Models**
```python
  history = model.fit(train_data, epochs=10, validation_data=test_data, shuffle=True)
```
9. **Model Evaluation on Validation Data**  
10. **Model Performance Visualization**  
11. **Output Prediction and Probability Estimation on Test Data**  

