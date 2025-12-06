import cv2
import numpy as np
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from time import time


def calcular_iou(box1, box2):
    """Calcula Intersection over Union entre dos cajas [x, y, w, h]."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x1_max = x1 + w1
    y1_max = y1 + h1
    x2_max = x2 + w2
    y2_max = y2 + h2

    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    union_area = (w1 * h1) + (w2 * h2) - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


# -------------------------------------------------------------------
# PROCESAMIENTO PRINCIPAL
# -------------------------------------------------------------------

TF_ENABLE_ONEDNN_OPTS = 0  # Desactiva optimizaciones si usas TensorFlow

# Cargar imagen
imagen = cv2.imread("Ejemplo.jpg")
rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

# Detector MTCNN
detector = MTCNN()

# Detección con temporizador
inicio = time()
detecciones = detector.detect_faces(rgb)
fin = time()

print(f"Rostros encontrados: {len(detecciones)} • Tiempo: {(fin - inicio) * 1000:.1f} ms")

for det in detecciones:
    print("Confianza:", det["confidence"])
    print("Caja:", det["box"])
    print("Puntos:", list(det["keypoints"].keys()))
    print("---")

# Dibujar detecciones
canvas = rgb.copy()
for det in detecciones:
    x, y, w, h = det["box"]
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for nombre, (px, py) in det["keypoints"].items():
        cv2.circle(canvas, (px, py), 3, (255, 0, 0), -1)

plt.imshow(canvas)
plt.axis("off")
plt.show()

# Filtrado por umbral de confianza
umbral = 0.90
validos = [d for d in detecciones if d["confidence"] >= umbral]
print(f"Con umbral={umbral}, quedan {len(validos)} detecciones")

# Ejemplo de IoU
if len(detecciones) > 1:
    ejemplo_iou = calcular_iou(detecciones[0]["box"], detecciones[-1]["box"])
    print("IoU entre primera y última detección:", ejemplo_iou)
