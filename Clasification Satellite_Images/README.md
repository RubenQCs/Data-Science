# 🛰️ Clasificación de Imágenes Satelitales con PyTorch  

Este proyecto implementa un modelo de **clasificación de imágenes satelitales** utilizando **PyTorch** y redes neuronales convolucionales (**CNNs**). Se emplea **ResNet-18** como modelo base para la extracción de características y entrenamiento supervisado.  

## 📌 Características del Proyecto  

✅ **Procesamiento de Datos:**  
- Descompresión y organización automática del conjunto de datos.  
- Creación de subconjuntos para entrenamiento, validación y prueba.  

✅ **Entrenamiento con Transfer Learning:**  
- Uso de **ResNet-18** preentrenada en ImageNet.  
- Congelación de capas iniciales y ajuste fino de la capa final.  

✅ **Evaluación del Modelo:**  
- Cálculo de métricas como **accuracy** y **F1-score**.  
- Visualización de la matriz de confusión.  

✅ **Visualización de Resultados:**  
- Gráficos de **pérdida y precisión** a lo largo de las épocas.  
- Análisis de la distribución de clases en el conjunto de datos.  

## 🔧 Tecnologías Utilizadas  

- **PyTorch** para el desarrollo del modelo.  
- **Torchvision** para transformaciones de imágenes.  
- **OpenCV y Matplotlib** para visualización de datos.  
- **Scikit-learn** para el análisis de métricas.  
