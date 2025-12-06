import cv2
from mtcnn.mtcnn import MTCNN
import time

detector = MTCNN()

cap = cv2.VideoCapture("Humano_estudiando.mp4")

prev = time.time()
frames = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)

    for f in faces:
        x, y, w, h = f["box"]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    frames += 1
    if frames % 10 == 0:
        now = time.time()
        fps = 10 / (now - prev)
        prev = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 👇 Reducción del tamaño de la ventana
    frame_resized = cv2.resize(frame, (640, 360))

    cv2.imshow("MTCNN Video", frame_resized)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()

# ---- WEBCAM ----

cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)

    for f in faces:
        x, y, w, h = f["box"]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    frame_resized = cv2.resize(frame, (640, 360))
    cv2.imshow("MTCNN Webcam", frame_resized)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
