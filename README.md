# 🚀 CodeLabs — Proyectos durante el segundo semestre 2025

Este repositorio reúne todos los ejercicios y mini-proyectos desarrollados durante los CodeLabs del semestre.  
Incluye visión por computador, reconocimiento de voz, redes neuronales y clasificación de texto.  
Todo está organizado para ejecutarse fácilmente en cualquier entorno.

---

## 📚 Contenido del repositorio

### 🎤 1. Reconocimiento de voz  
- Captura de audio desde micrófono.  
- Procesamiento y transcripción usando librerías modernas.  
- Ejecución en consola.

---

### 😊 2. MTCNN — Detección de rostros en imágenes  
- Detección de rostros con MTCNN.  
- Dibujar bounding boxes y puntos faciales.  
- Soporte para imágenes locales.

---

### 🎥 3. MTCNN — Detección de rostros en video  
- Lectura desde webcam.  
- Inferencia en tiempo real.  
- Detección por cuadro con MTCNN.

---

### ✖️ 4. Red neuronal XOR (Perceptrón)  
- Implementación desde cero con Keras / TensorFlow.  
- Entrenamiento de una red capaz de aprender XOR.  
- Predicciones para todas las combinaciones binarias.

---

### 🟧 5. YOLO Lite — Detección de objetos en imágenes  
- Uso de modelo YOLOv8 nano (`yolov8n.pt`).  
- Inferencia rápida sobre imágenes.  
- Visualización de clases y cajas.

---

### 🟩 6. YOLOv8 — Detección de objetos en webcam  
- Lectura desde cámara.  
- Inferencia en tiempo real con Ultralytics.  
- Cierre con tecla **q**.  

---

### 🟦 7. SSD300-VGG16 — Detección de objetos  
- Uso de `ssd300_vgg16` con pesos COCO.  
- Preprocesamiento oficial con transforms.  
- Medición de tiempo de inferencia.  
- Dibujar boxes y etiquetas.

---

### 💬 8. Clasificador de comentarios (NLP)  
- Limpieza y normalización de texto.  
- Vectorización con TF-IDF.  
- Clasificador **Linear SVC**.  
- Validación cruzada (5-fold).  
- Reportes (accuracy, F1, matriz de confusión).  
- Guardado y carga del modelo (`modelo.joblib` + `tfidf.joblib`).

---

## 🧩 Dependencias principales

```bash
pip install torch torchvision torchaudio
pip install ultralytics
pip install opencv-python
pip install tensorflow
pip install scikit-learn pandas numpy matplotlib joblib
pip install roboflow
```

## 🧪 Entorno virtual

```bash
# Crear
python -m venv .venv

# Activar (Windows)
.venv\Scripts\activate

# Activar (Linux / Mac)
source .venv/bin/activate

# Desactivar
deactivate
```

## 📝 Licencia

Este proyecto es solo para fines académicos y demostrativos.


