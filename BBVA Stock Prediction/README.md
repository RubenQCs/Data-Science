# 📈[In Progress...] BBVA Stock Price Prediction with Machine Learning
**Note:** This project is currently **in progress**.

This repository contains a project aimed at **predicting the stock price of BBVA** using **Machine Learning** techniques. The goal is to analyze historical BBVA data and build models to estimate its future price movements.

## 🔍 **What’s in this repository?**  
- Downloading historical BBVA data.
- Exploratory Data Analysis (EDA) and trend visualization.  
- Application of Machine Learning models. 
- Evaluation of model performance using metrics like **RMSE**.  
- Predictions and result visualization.  

## 🛠 **Technologies used**  
- Python 
- `yfinance` for financial data download.  
- `pandas` and `numpy` for data manipulation.  
- `matplotlib` and `seaborn` for visualization.  
- `scikit-learn` and `TensorFlow/PyTorch` for Machine Learning models.  


## 📌 General Workflow  

1. **import`yfinance` for financial data download**

```Python
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# Descargar datos históricos de BBVA (Bolsa de Madrid)
bbva = yf.Ticker("BBVA.MC")
hist = bbva.history(period="max")

# Seleccionar solo las columnas requeridas
hist = hist[['Open', 'High', 'Low', 'Close', 'Volume']]
hist.reset_index(inplace=True)  # Convertir índice (fecha) en columna normal

#Formatear la columna Date para que solo muestre "YYYY-MM-DD"
hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')

# Redondear valores a 2 decimales
hist[['Open', 'High', 'Low', 'Close']] = hist[['Open', 'High', 'Low', 'Close']].round(2)

# Mostrar las últimas 10 filas
hist_tail = hist.tail(10)

# Mostrar tabla en consola
print("Datos históricos de BBVA (últimos 10 registros):")
print(hist_tail)

# Guardar tabla como imagen
fig, ax = plt.subplots(figsize=(10, 3))  # Crear figura
ax.axis('tight')
ax.axis('off')  # Ocultar ejes
table_data = hist_tail.values  # Datos sin encabezados
column_labels = hist_tail.columns  # Encabezados

# Crear tabla 
table = ax.table(cellText=table_data, colLabels=column_labels, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)  # Ajustar tamaño

# Guardar tabla en JPG
plt.savefig("bbva_tabla_historica.jpg", format="jpg", dpi=300, bbox_inches="tight")
plt.close()

print("Tabla guardada como 'bbva_tabla_historica.jpg'")

# Gráfico del precio del stock y Volumen
# Convertir la columna 'Date' a formato datetime
hist['Date'] = pd.to_datetime(hist['Date'])

# Crear una figura con dos subgráficas (2 columnas)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))  # 1 fila, 2 columnas

# Gráfico del precio histórico vs fecha (Figura 1)
ax1.plot(hist["Date"], hist["Close"], label="Precio de Cierre", color="b", linewidth=0.8)
ax1.set_xlabel("Año")
ax1.set_ylabel("Precio (EUR)")
ax1.set_title("Evolución Histórica del Precio de BBVA")
ax1.legend()
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
ax1.xaxis.set_major_locator(mdates.YearLocator(5))  # Ticks mayores cada 5 años
ax1.xaxis.set_minor_locator(mdates.YearLocator(1))  # Ticks menores cada 1 año
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Formato de año

# Gráfico del volumen histórico vs fecha (Figura 2)
ax2.plot(hist["Date"], hist["Volume"], label="Volumen", color="g", linewidth=0.8)
ax2.set_xlabel("Año")
ax2.set_ylabel("Volumen de Acción")
ax2.set_title("Volumen de Acciones de BBVA")
ax2.legend()
ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
ax2.xaxis.set_major_locator(mdates.YearLocator(5))  # Ticks mayores cada 5 años
ax2.xaxis.set_minor_locator(mdates.YearLocator(1))  # Ticks menores cada 1 año
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Formato de año

# Ajustar los espacios entre las subgráficas
plt.tight_layout()

# Guardar el gráfico en JPG
plt.savefig("bbva_precio_y_volumen_historico.jpg", format='jpg', dpi=300)

# Mostrar los gráficos
plt.show()

print("Gráfico guardado como 'bbva_precio_y_volumen_historico.jpg'")

```
![Plot](plots/bbva_precio_y_volumen_historico.jpg)
![Table I](plots/bbva_tabla_historica.jpg)

2. **Display descriptive statistics and information about the data**

```Python
desc_stats = hist.describe()

# Print statistics to the console
print("Descriptive Statistics for BBVA Stock Data:")
print(desc_stats)


#  Convert DataFrame to string format for display
desc_stats_rounded = desc_stats.round(3)  # Round values for better readability
desc_stats_str = desc_stats_rounded.astype(str)

#  Plot and save table as JPG
fig, ax = plt.subplots(figsize=(10, 4))  # Adjust figure size
ax.axis('tight')
ax.axis('off')  # Hide axes

# crear tabla
table = ax.table(cellText=desc_stats_str.values, 
                 colLabels=desc_stats_str.columns, 
                 rowLabels=desc_stats_str.index, 
                 cellLoc='center', loc='center')

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)  # Adjust size

# guardar JPG
plt.savefig("bbva_descriptive_statistics.jpg", format="jpg", dpi=300, bbox_inches="tight")
plt.close()

print("Table saved as 'bbva_descriptive_statistics.jpg'.")
```

![Table II](plots/bbva_descriptive_statistics.jpg)

```Python
# Información de los datos
hist.info()
print("Table saved as 'bbva_data_info.jpg'.")
```
![Table IV](plots/BBVA_info_data.jpg)




2. **Moving average**

```Python

# Calcular las medias móviles para 10, 20 y 50 días
ma_days = [10, 20, 50]
for ma in ma_days:
    hist[f'MA {ma}'] = hist['Close'].rolling(ma).mean()

# Definir el período de zoom (últimos 365 días)
hist_zoom = hist.tail(365)

# Crear la figura principal
fig, ax1 = plt.subplots(figsize=(12, 6))

# Gráfico principal: Precio y medias móviles
ax1.plot(hist["Date"], hist["Close"], label="Precio de Cierre", color="b", linewidth=1)
ax1.plot(hist["Date"], hist["MA 10"], label="MA 10 días", color="r", linewidth=0.8)
ax1.plot(hist["Date"], hist["MA 20"], label="MA 20 días", color="g", linewidth=0.8)
ax1.plot(hist["Date"], hist["MA 50"], label="MA 50 días", color="orange", linewidth=0.8)

# Configuración del gráfico principal
ax1.set_xlabel("Año")
ax1.set_ylabel("Precio (EUR)")
ax1.set_title("Evolución Histórica del Precio de BBVA y Medias Móviles")
ax1.legend()
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
ax1.xaxis.set_major_locator(mdates.YearLocator(5))
ax1.xaxis.set_minor_locator(mdates.YearLocator(1))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Dibujar líneas del Zoom
ax1.axvline(hist_zoom["Date"].iloc[0], color="gray", linestyle="--", linewidth=1.2)
ax1.axvline(hist_zoom["Date"].iloc[-1], color="gray", linestyle="--", linewidth=1.2)

# Subplot de zoom 
ax_zoom = fig.add_axes([0.45, 0.6, 0.3, 0.3])  # [left, bottom, width, height]
ax_zoom.plot(hist_zoom["Date"], hist_zoom["Close"], label="Precio de Cierre", color="b", linewidth=1.2)
ax_zoom.plot(hist_zoom["Date"], hist_zoom["MA 10"], color="r", linewidth=0.8)
ax_zoom.plot(hist_zoom["Date"], hist_zoom["MA 20"], color="g", linewidth=0.8)
ax_zoom.plot(hist_zoom["Date"], hist_zoom["MA 50"], color="orange", linewidth=0.8)
ax_zoom.set_title("Zoom: Últimos 365 días", fontsize=10)
ax_zoom.tick_params(axis='both', labelsize=8)
ax_zoom.grid(True, linestyle='--', linewidth=0.5)

# Configurar el eje X del zoom 
ax_zoom.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(ax_zoom.xaxis.get_majorticklabels(), rotation=60)
  # Etiquetas cada 2 meses
ax_zoom.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Formato Año-Mes

# Ajustar la disposición de la figura
plt.tight_layout()

# Guardar el gráfico en JPG
plt.savefig("bbva_precio_con_ma_zoom_centrado.jpg", format='jpg', dpi=300, bbox_inches="tight")

# Mostrar los gráficos
plt.show()

print("Gráfico guardado como 'bbva_precio_con_ma_zoom_centrado.jpg'")

```
![Moving average](plots/bbva_precio_con_ma_zoom_centrado.jpg)
