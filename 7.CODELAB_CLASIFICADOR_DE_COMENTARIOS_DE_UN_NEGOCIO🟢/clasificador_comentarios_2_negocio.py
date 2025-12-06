# ----------------------------------------
# Requisitos:
# pip install scikit-learn pandas numpy joblib matplotlib
# ----------------------------------------

import re
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import make_pipeline

import joblib

# ========================================
# 1) Dataset sintético realista
# ========================================

random.seed(42)
np.random.seed(42)

positivos = [
    "Excelente servicio","Muy buena atención","Me encantó el producto",
    "Rápido y confiable","Todo llegó perfecto","Calidad superior",
    "Lo recomiendo totalmente","Volveré a comprar","Precio justo y buena calidad",
    "El soporte fue amable","Experiencia increíble","Funcionó mejor de lo esperado",
    "Entregado a tiempo","Muy satisfecho","Cinco estrellas",
    "La comida estaba deliciosa","El empaque impecable","Súper recomendable",
    "Buen trato del personal","Gran experiencia"
]

negativos = [
    "Pésimo servicio","Muy mala atención","Odio este producto",
    "Lento y poco confiable","Llegó dañado","Calidad terrible",
    "No lo recomiendo","No vuelvo a comprar","Caro y mala calidad",
    "El soporte fue grosero","Experiencia horrible","Peor de lo esperado",
    "Entregado tarde","Muy decepcionado","Una estrella",
    "La comida estaba fría","El empaque roto","Nada recomendable",
    "Mal trato del personal","Mala experiencia"
]

def variantes(frase):
    extras = [
        "", "!", "!!", " 🙂", " 😡", " de verdad", " en serio", 
        " 10/10", " 1/10", " súper", " la verdad", " jamás", " nunca", 
        " para nada"
    ]
    return frase + random.choice(extras)

# Expandir dataset
pos = [variantes(p) for _ in range(8) for p in positivos]
neg = [variantes(n) for _ in range(8) for n in negativos]

textos = pos + neg
etiquetas = [1]*len(pos) + [0]*len(neg)

df = pd.DataFrame({"texto": textos, "etiqueta": etiquetas})
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Muestras:", len(df), 
      " | Positivos:", df.etiqueta.sum(), 
      " | Negativos:", len(df)-df.etiqueta.sum())

# ========================================
# 2) Limpieza simple
# ========================================

def limpiar(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-záéíóúñü0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

df["texto_clean"] = df["texto"].apply(limpiar)

# ========================================
# 3) Train/Test split + Baseline
# ========================================

X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["texto_clean"], df["etiqueta"],
    test_size=0.2, random_state=42, stratify=df["etiqueta"]
)

mayoritaria = int(round(y_train.mean()))
baseline = (y_test == mayoritaria).mean()
print(f"Baseline (clase mayoritaria): {baseline:.3f}")

# ========================================
# 4) Vectorización TF-IDF
# ========================================

vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1,2),
    min_df=2
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# ========================================
# 5) Modelo Linear SVC
# ========================================

clf = LinearSVC(class_weight="balanced", random_state=42)
clf.fit(X_train, y_train)

# ========================================
# 6) Evaluación
# ========================================

pred = clf.predict(X_test)
acc = accuracy_score(y_test, pred)

print(f"\nAccuracy test: {acc:.3f}  |  Mejora vs baseline: {acc - baseline:.3f}\n")
print("Reporte por clase:")
print(classification_report(y_test, pred, digits=3))

cm = confusion_matrix(y_test, pred)
print("\nMatriz de confusión:")
print(pd.DataFrame(
    cm,
    index=["Real 0 (neg)", "Real 1 (pos)"],
    columns=["Pred 0 (neg)", "Pred 1 (pos)"]
))

# ========================================
# 7) Cross Validation 5-fold
# ========================================

pipe = make_pipeline(
    TfidfVectorizer(max_features=30000, ngram_range=(1,2), min_df=2),
    LinearSVC(class_weight="balanced", random_state=42)
)

scores = cross_val_score(pipe, df["texto_clean"], df["etiqueta"], cv=5, scoring="f1_macro")
print(f"\nCV (5-fold) F1_macro: {scores.mean():.3f} ± {scores.std():.3f}")

# ========================================
# 8) Predicciones nuevas
# ========================================

def predecir(textos_nuevos):
    tx = [limpiar(t) for t in textos_nuevos]
    Xn = vectorizer.transform(tx)
    p = clf.predict(Xn)
    return ["positivo" if i==1 else "negativo" for i in p]

nuevos = [
    "El envío fue rapidísimo y el empaque llegó impecable, gracias!",
    "Demoraron demasiado y además nadie respondió los mensajes",
    "Calidad/precio brutal, quedé muy satisfecho",
    "No lo recomiendo, salió defectuoso y me tocó devolverlo"
]

print("\nPredicciones nuevas:")
for t, etiqueta in zip(nuevos, predecir(nuevos)):
    print(f"- {t}  ->  {etiqueta}")

# ========================================
# 9) Guardar modelo
# ========================================

joblib.dump(vectorizer, "tfidf.joblib")
joblib.dump(clf, "modelo.joblib")
print("\nModelo y vectorizador guardados.")

# Verificación
vec = joblib.load("tfidf.joblib")
model = joblib.load("modelo.joblib")

X_prueba = vec.transform(["La compra fue excelente, todo perfecto"])
print("Pred loaded model:", 
      "positivo" if model.predict(X_prueba)[0]==1 else "negativo")
