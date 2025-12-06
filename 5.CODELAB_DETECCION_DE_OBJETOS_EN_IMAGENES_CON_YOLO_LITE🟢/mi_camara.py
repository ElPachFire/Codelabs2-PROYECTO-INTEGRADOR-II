import cv2
from ultralytics import YOLO

# ------------------------------
# Cargar modelo YOLOv8 Nano
# ------------------------------
modelo = YOLO("yolov8n.pt")

# ------------------------------
# Activar webcam
# ------------------------------
camara = cv2.VideoCapture(0)

while True:
    ok, frame = camara.read()
    if not ok:
        break

    # Procesar frame
    detecciones = modelo(frame)
    frame_etiquetado = detecciones[0].plot()

    # Mostrar ventana
    cv2.imshow("YOLOv8 - Webcam", frame_etiquetado)

    # Tecla 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camara.release()
cv2.destroyAllWindows()
