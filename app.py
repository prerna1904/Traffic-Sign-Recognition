from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

model = load_model("models/traffic_sign_model.keras")

IMG_HEIGHT = 64
IMG_WIDTH = 64

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    filepath = "temp.jpg"
    file.save(filepath)

    img = image.load_img(filepath, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    os.remove(filepath)

    return jsonify({
        "label": f"Class {predicted_class}",
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True)
