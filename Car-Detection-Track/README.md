# 🚗 Vehicle Detection, Tracking, and Counting with YOLO  

This project implements a **vehicle detection, tracking, and counting system** using **YOLO (You Only Look Once)**, an advanced real-time object detection model.  

## 🔹 Project Description  
The goal of this project is to develop an efficient solution for traffic analysis, capable of:  

**Detecting vehicles in images or videos** using YOLO.  
 **Tracking each vehicle** as it moves through the scene.  
 **Counting detected vehicles**.  

This system can be applied in various scenarios, such as **traffic monitoring, toll management, vehicle flow analysis, and smart cities**.  

## Technologies Used  
- **YOLOv8** for real-time detection.  
- **DeepSORT** for vehicle tracking.  
- **OpenCV** for video processing.  
- **Python** as the main programming language.  

```Python
import datetime
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Configuración del modelo YOLO
MODEL_PATH = "yolo8n.pt"
CONFIDENCE_THRESHOLD = 0.7
CLASS_NAME = "car"
VIDEO_PATH = "example/Car6.mp4"

# Cargar modelo YOLO
model = YOLO(MODEL_PATH)

# Inicializar el rastreador Deep SORT
tracker = DeepSort(max_age=50)

# Inicializar captura de video
cap = cv2.VideoCapture(VIDEO_PATH)
object_counter = set()

while cap.isOpened():
    start_time = datetime.datetime.now()
    success, frame = cap.read()
    
    if not success:
        break
    
    # Ejecutar YOLO en el frame
    results = model(frame)
    boxes = results[0].boxes.xyxy.cpu()
    class_ids = results[0].boxes.cls.cpu()
    confidences = results[0].boxes.conf.cpu()
    
    filtered_detections = []
    for confidence, box, class_id in zip(confidences, boxes, class_ids):
        if float(confidence) >= CONFIDENCE_THRESHOLD and model.names[int(class_id)] == CLASS_NAME:
            xmin, ymin, xmax, ymax = map(int, box)
            filtered_detections.append([[xmin, ymin, xmax - xmin, ymax - ymin], float(confidence), int(class_id)])
    
    # Actualizar rastreador con detecciones filtradas
    tracks = tracker.update_tracks(filtered_detections, frame=frame)
    
    for track in tracks:
        if not track.is_confirmed():
            continue
        
        track_id = track.track_id
        object_counter.add(track_id)
        xmin, ymin, xmax, ymax = map(int, track.to_ltrb())
        
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 255), 2)
        cv2.rectangle(frame, (xmin, ymin - 30), (xmin + 30, ymin), (255, 255, 255), -1)
        cv2.putText(frame, str(track_id), (xmin + 5, ymin - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
    cv2.putText(frame, f"Objects detected: {len(object_counter)}", (1500, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
    
    processing_time = (datetime.datetime.now() - start_time).total_seconds()
    fps = f"FPS: {1 / processing_time:.2f}"
    cv2.putText(frame, fps, (1500, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
    
    cv2.imshow("YOLO11 Tracking", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()

```

## 🎥 Ejemplo de Detección y Tracking  

📌 **Ver el video en Google Drive:**  
[📹 Ver Video en Google Drive](https://drive.google.com/open?id=1NXBTsCLbZFCL07NkiOg5n1tSnz1lcPG2)
