# 🚀 CodeLabs de Visión por Computador & NLP — Proyecto Completo

Este repositorio reúne todos los ejercicios y mini-proyectos realizados durante los CodeLabs, incluyendo visión por computador, clasificación de texto, redes neuronales y uso de datasets externos.  
Todo está organizado para que pueda ejecutarse fácilmente en cualquier entorno.

---

## 📚 Contenido del repositorio

### 🟦 1. Detección facial con MTCNN / dlib / OpenCV
- Lectura desde webcam.
- Detección de rostros.
- Dibujar cajas y puntos.
- Ajustes de tamaño de ventana.

### 🟩 2. Detección de objetos con SSD300-VGG16 (PyTorch)
- Uso de `SSD300_VGG16_Weights`.
- Preprocesamiento con transforms oficiales.
- Visualización de bounding boxes.
- Medición del tiempo de inferencia.

### 🟧 3. YOLOv8 en webcam (Ultralytics)
- Modelo `yolov8n.pt` para detección rápida.
- Anotación automática de resultados.
- Ejecución en tiempo real.
- Salida con tecla `q`.

### 🟪 4. Descarga de datasets con Roboflow API
- Lectura de API key desde variables de entorno.
- Acceso a Workspaces y Projects.
- Descarga automática de datasets (YOLO, COCO, etc.).

### 🟨 5. Clasificador de comentarios (NLP)
- Dataset sintético aumentado (positivo / negativo).
- Limpieza básica de texto.
- TF-IDF con unigramas y bigramas.
- Clasificador **Linear SVC**.
- Validación cruzada (5-fold).
- Reportes: accuracy, F1, matriz de confusión.
- Guardado del modelo (`modelo.joblib` y `tfidf.joblib`).
- Predicción con modelos cargados.

### 🟫 6. Red neuronal XOR con Keras
- Red simple: capa oculta ReLU + salida Sigmoid.
- Entrenamiento de 5000 épocas.
- Predicciones para todas las combinaciones XOR.

---


## 🧩 Dependencias principales

Instalar todo con:

```bash
pip install torch torchvision torchaudio
pip install ultralytics
pip install opencv-python
pip install tensorflow
pip install scikit-learn pandas numpy matplotlib joblib
pip install roboflow


🧪 Entorno virtual
Crear
python -m venv .venv

Activar

Windows

.venv\Scripts\activate


Linux / Mac

source .venv/bin/activate

Desactivar
deactivate
```

## 📝 Licencia

Este proyecto es solo para fines académicos y demostrativos.


