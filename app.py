import os
import cv2
import numpy as np
from flask import Flask, render_template, request
from tensorflow import keras

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "wpb_model_opti.keras")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

model = keras.models.load_model(MODEL_PATH, compile=False)
model.trainable = False

# Define class labels
class_labels = {0: "none", 1: "target"}

# Prediction Function
def predict_image(image_path):
    # Load and preprocess image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224)) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0).astype(np.float32)

    # Get model prediction (directly return 0 or 1)
    predicted_folder = int(model.predict(img, verbose=0).round())

    # Get confidence score
    confidence = model.predict(img, verbose=0)[0][0]

    # Get class label
    predicted_label = class_labels[predicted_folder]

    return predicted_label, confidence * 100  # Return class name & confidence %

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle File Upload
        if "file" not in request.files:
            return render_template("index.html", message="No file uploaded.")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", message="No selected file.")

        # Save the uploaded file
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        img_relative_path = os.path.join("static", "uploads", file.filename)

        # Classify Image
        predicted_class, confidence = predict_image(file_path)

        return render_template("index.html", uploaded_file=img_relative_path, bird=predicted_class, confidence=confidence)

    return render_template("index.html")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)