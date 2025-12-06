from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import time

# ------------------------------------------------------
# Cargar modelo SSD300-VGG16 con pesos preentrenados
# ------------------------------------------------------
pesos = SSD300_VGG16_Weights.DEFAULT
modelo = ssd300_vgg16(weights=pesos).eval()

# Transformaciones oficiales (resize + normalización)
transformar = pesos.transforms()

# ------------------------------------------------------
# Cargar imagen
# ------------------------------------------------------
imagen = Image.open("gato.jpeg").convert("RGB")
tensor = transformar(imagen).unsqueeze(0)

# ------------------------------------------------------
# Inferencia
# ------------------------------------------------------
with torch.no_grad():
    inicio = time.time()
    salida = modelo(tensor)[0]
    fin = time.time()

print(f"Tiempo SSD: {fin - inicio:.4f} s")

# ------------------------------------------------------
# Extraer predicciones
# ------------------------------------------------------
cajas = salida["boxes"]
etiquetas = salida["labels"]
puntuaciones = salida["scores"]
clases = pesos.meta["categories"]

# ------------------------------------------------------
# Visualizar resultados
# ------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(imagen)

for box, lab, score in zip(cajas, etiquetas, puntuaciones):
    if score < 0.5:
        continue

    x1, y1, x2, y2 = box.tolist()
    
    ax.add_patch(
        patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="red",
            facecolor="none"
        )
    )

    ax.text(
        x1,
        y1,
        f"{clases[int(lab)]}: {score:.2f}",
        color="white",
        fontsize=8,
        bbox=dict(facecolor="black", alpha=0.5)
    )

plt.axis("off")
plt.show()
