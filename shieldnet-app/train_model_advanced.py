# train_model_advanced.py
"""
Entrenamiento avanzado para ShieldNet:
- Carga dataset.csv (columnas: text,label)
- Preprocesa texto básico
- Prueba LogisticRegression (GridSearch) y MLPClassifier (GridSearch)
- Selecciona el mejor modelo por validación cruzada
- Calibra probabilidades con CalibratedClassifierCV
- Guarda pipeline final a model_shieldnet_advanced.joblib
"""

import os
import re
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight

# Configuración
RANDOM_STATE = 42
MODEL_OUTPATH = "model_shieldnet_advanced.joblib"
DATA_PATH = "dataset.csv"

# 1) Cargar datos
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"No encontrado {DATA_PATH} en el directorio actual: {os.getcwd()}")

df = pd.read_csv(DATA_PATH)
# Asegurar columnas correctas
if "text" not in df.columns or "label" not in df.columns:
    raise ValueError("dataset.csv debe tener columnas 'text' y 'label'")

# 2) Limpieza simple del texto
def clean_text(s):
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"http\S+", " ", s)                # eliminar URLs
    s = re.sub(r"@\w+", " ", s)                   # eliminar menciones
    s = re.sub(r"[^a-záéíóúñü0-9\s]", " ", s)     # mantener letras (incl. acentos) y números
    s = re.sub(r"\s+", " ", s).strip()
    return s

df["text_clean"] = df["text"].astype(str).apply(clean_text)
X = df["text_clean"].values
y = df["label"].values

# 3) División entrenamiento / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE
)

print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")

# 4) Calcular pesos de clase si hay desbalance (opcional)
classes = np.unique(y_train)
class_weights = dict(zip(classes, compute_class_weight(class_weight="balanced", classes=classes, y=y_train)))

print("Pesos de clase:", class_weights)

# 5) Pipeline TF-IDF + modelo (probamos dos familias: LogisticRegression y MLP)
tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=2)

# Logistic Regression (rápido, robusto)
pipe_lr = Pipeline([
    ("tfidf", tfidf),
    ("clf", LogisticRegression(max_iter=2000, solver="saga", random_state=RANDOM_STATE))
])

param_grid_lr = {
    "clf__C": [0.01, 0.1, 1, 5],
    "clf__penalty": ["l2", "l1"],
    "tfidf__ngram_range": [(1,1), (1,2)]
}

# MLP (red neuronal pequeña)
pipe_mlp = Pipeline([
    ("tfidf", tfidf),
    ("clf", MLPClassifier(random_state=RANDOM_STATE, max_iter=1000))
])

param_grid_mlp = {
    "clf__hidden_layer_sizes": [(50,), (100,), (100,50)],
    "clf__alpha": [1e-4, 1e-3],
    "tfidf__ngram_range": [(1,1), (1,2)]
}

# Validación cruzada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# 6) GridSearch para LR
print("Entrenando LogisticRegression con GridSearchCV...")
gs_lr = GridSearchCV(pipe_lr, param_grid_lr, scoring="accuracy", cv=cv, n_jobs=-1, verbose=1)
gs_lr.fit(X_train, y_train)
print("Mejor LR:", gs_lr.best_params_, "CV score:", gs_lr.best_score_)

# 7) GridSearch para MLP
print("Entrenando MLPClassifier con GridSearchCV...")
gs_mlp = GridSearchCV(pipe_mlp, param_grid_mlp, scoring="accuracy", cv=cv, n_jobs=-1, verbose=1)
gs_mlp.fit(X_train, y_train)
print("Mejor MLP:", gs_mlp.best_params_, "CV score:", gs_mlp.best_score_)

# 8) Seleccionar el mejor modelo según score en validación
best_model = gs_lr.best_estimator_ if gs_lr.best_score_ >= gs_mlp.best_score_ else gs_mlp.best_estimator_
best_name = "LogisticRegression" if gs_lr.best_score_ >= gs_mlp.best_score_ else "MLPClassifier"
print("Modelo seleccionado:", best_name)

# 9) Calibrar probabilidades (mejora interpretabilidad de predict_proba)
print("Calibrando probabilidades con CalibratedClassifierCV (Platt)...")
calibrator = CalibratedClassifierCV(best_model, cv=cv, method="sigmoid")
calibrator.fit(X_train, y_train)

# 10) Evaluación en test
y_pred = calibrator.predict(X_test)
y_proba = calibrator.predict_proba(X_test)

print("\n--- Evaluación en test set ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# 11) Guardar pipeline calibrado junto con vectorizador dentro de joblib
# Guardamos el calibrator (contiene el pipeline con tfidf dentro)
joblib.dump(calibrator, MODEL_OUTPATH)
print(f"Modelo final calibrado guardado en: {MODEL_OUTPATH}")

# 12) Ejemplo de uso rápido (sanity check)
def predict_text(text):
    cleaned = clean_text(text)
    pred = calibrator.predict([cleaned])[0]
    proba = calibrator.predict_proba([cleaned])[0].max()
    return pred, float(proba)

examples = [
    "Eres un idiota, me caes fatal",
    "Muchas gracias por la ayuda, excelente trabajo"
]
for e in examples:
    p, prob = predict_text(e)
    print(f"Ejemplo: {e} -> {p} ({prob:.2f})")
