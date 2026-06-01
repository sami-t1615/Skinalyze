import sqlite3
from datetime import datetime
import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from PIL import Image

app = Flask(__name__)

# ==========================
# LOAD MODEL
# ==========================
model = tf.keras.models.load_model("skin_mobilenetv2.h5")

# Load class labels
with open("class_indices.json") as f:
    class_indices = json.load(f)

# Reverse mapping
idx_to_class = {v: k for k, v in class_indices.items()}

IMG_SIZE = 224

# ==========================
# DATABASE INIT
# ==========================
def init_db():
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        age INTEGER,
        gender TEXT,
        prediction TEXT,
        confidence REAL,
        date TEXT
    )
    """)

    conn.commit()
    conn.close()

init_db()

# ==========================
# PREPROCESS IMAGE
# ==========================
def preprocess(file):
    img = Image.open(file).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ==========================
# ROUTES
# ==========================
@app.route("/")
def home():
    return render_template("index.html")

# 🔥 PREDICT
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img = preprocess(request.files["image"])
    preds = model.predict(img)[0]

    result = {}
    for i, prob in enumerate(preds):
        result[idx_to_class[i]] = float(prob)

    return jsonify(result)

# 💾 SAVE TO DATABASE
@app.route("/save", methods=["POST"])
def save():
    data = request.json

    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO history (name, age, gender, prediction, confidence, date)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (
        data.get("name"),
        data.get("age"),
        data.get("gender"),
        data.get("prediction"),
        data.get("confidence"),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()

    return jsonify({"message": "Saved successfully"})

# 📜 GET HISTORY
@app.route("/history", methods=["GET"])
def get_history():
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM history ORDER BY id DESC")
    rows = cursor.fetchall()

    conn.close()

    history = []
    for row in rows:
        history.append({
            "id": row[0],
            "name": row[1],
            "age": row[2],
            "gender": row[3],
            "prediction": row[4],
            "confidence": row[5],
            "date": row[6]
        })

    return jsonify(history)

# ==========================
# RUN APP
# ==========================
if __name__ == "__main__":
    app.run(debug=True)