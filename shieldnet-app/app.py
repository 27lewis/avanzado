# app.py
from flask import Flask, request, jsonify, render_template
import joblib
import os
from datetime import datetime
from sqlalchemy import create_engine, text

# ==========================
# CONFIGURACIÓN BASE DE DATOS (XAMPP / MySQL)
# ==========================
DB_USER = "root"
DB_PASSWORD = ""   # deja vacío si tu MySQL no tiene contraseña
DB_HOST = "127.0.0.1"
DB_PORT = "3306"
DB_NAME = "shieldnet"

DATABASE_URI = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"

engine = create_engine(DATABASE_URI, pool_recycle=3600)

# ==========================
# FLASK CONFIG
# ==========================
app = Flask(__name__, template_folder="templates", static_folder="static")

# ==========================
# CARGAR MODELO AVANZADO
# ==========================
MODEL_PATH = "model_shieldnet_advanced.joblib"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"No se encontró el modelo avanzado en {MODEL_PATH}. Ejecuta train_model_advanced.py primero.")

model = joblib.load(MODEL_PATH)

print("✅ Modelo avanzado cargado correctamente:", MODEL_PATH)

# ==========================
# RUTA PRINCIPAL (HTML)
# ==========================
@app.route("/")
def index():
    return render_template("index.html")

# ==========================
# ENDPOINT /api/analyze
# ==========================
@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json() or {}
    text_input = data.get("text") or data.get("text_input") or ""

    if not text_input:
        return jsonify({"error": "El campo 'text' es obligatorio"}), 400

    # ======= PREDICCIÓN =======
    try:
        probs = model.predict_proba([text_input])[0]
        classes = model.classes_
        max_idx = probs.argmax()
        label = classes[max_idx]
        confidence = float(probs[max_idx])
    except Exception as e:
        return jsonify({"error": "Error al procesar el texto", "detail": str(e)}), 500

    # ======= GUARDAR EN BASE DE DATOS =======
    try:
        with engine.connect() as conn:
            stmt = text("""
                INSERT INTO analyses (text_input, label, confidence, created_at)
                VALUES (:t, :l, :c, :d)
            """)
            conn.execute(stmt, {"t": text_input, "l": label, "c": confidence, "d": datetime.now()})
            conn.commit()
    except Exception as e:
        print("⚠️ Error al guardar en DB:", e)

    # ======= RESPUESTA =======
    return jsonify({
        "text": text_input,
        "label": label,
        "confidence": confidence
    })

# ==========================
# EJECUCIÓN LOCAL
# ==========================
if __name__ == "__main__":
    app.run(debug=True, port=5000)
