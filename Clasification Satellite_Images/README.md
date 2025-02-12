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
