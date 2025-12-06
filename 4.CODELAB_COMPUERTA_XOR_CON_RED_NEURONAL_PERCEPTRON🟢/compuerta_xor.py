import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# -------------------------
# Datos XOR
# -------------------------
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# -------------------------
# Modelo
# -------------------------
model = Sequential([
    Input(shape=(2,)),
    Dense(4, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# -------------------------
# Entrenamiento
# -------------------------
model.fit(X, y, epochs=5000, verbose=0)

# -------------------------
# Evaluación
# -------------------------
loss, acc = model.evaluate(X, y, verbose=0)
print(f"\nResultado final → Loss: {loss:.3f} | Accuracy: {acc:.3f}\n")

# -------------------------
# Predicciones XOR
# -------------------------
for a, b in X:
    pred = model.predict(np.array([[a, b]]), verbose=0)
    print(f"{a} XOR {b} = {round(pred.item(), 3)}")
