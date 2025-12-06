import torch
from PIL import Image
from torchvision import models, transforms

# -------------------------
# Cargar modelo preentrenado
# -------------------------
detector = models.detection.ssd300_vgg16(pretrained=True)
detector.eval()  # modo inferencia

# -------------------------
# Transformación para SSD300
# -------------------------
preprocesar = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor()
])

# -------------------------
# Cargar y preparar imagen
# -------------------------
imagen = Image.open("gatos.jpg")
tensor_img = preprocesar(imagen).unsqueeze(0)  # batch de 1

# -------------------------
# Inferencia (sin gradientes)
# -------------------------
with torch.no_grad():
    resultado = detector(tensor_img)

# -------------------------
# Mostrar detecciones crudas
# -------------------------
print(resultado)
