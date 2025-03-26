import cv2
import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow import keras
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "wpb_model_opti.keras")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

model = keras.models.load_model(MODEL_PATH, compile=False)
model.trainable = False

# Define class labels
class_labels = {0: "none", 1: "target"}


@app.route('/test', methods=['GET'])
def test():
    name = request.args.get('name')
    return jsonify({
        "status": "success",
        "attack": name
    })



#=========== URL base Prediction Start
# predict using image url
@app.route('/predict', methods=['GET'])
def predict():
    image_url = request.args.get('img_url')

    if not image_url:
        return jsonify({"error": "Missing 'url' parameter"}), 400

    # Save in UPLOAD_FOLDER
    filename = "temp_img.jpg"
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    success, message = temp_save(image_url, save_path)

    # Classify Image
    predicted_class, confidence = predict_image(save_path)
    confidence_rate = f"{confidence:.2f}%"

    # remove temp file
    os.remove(save_path)

    if success:
        return jsonify({
            "status": "success",
            "attack": predicted_class,
            "confidence": confidence_rate
        })
    else:
        return jsonify({"status": "error", "message": message}), 500

# Temp save url image for prediction
def temp_save(url, save_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise error for HTTP issues

        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)

        return True, f"Image saved to {save_path}"
    except requests.exceptions.RequestException as e:
        return False, f"Failed to download image: {str(e)}"
    except IOError as e:
        return False, f"Failed to save image: {str(e)}"

#=========== URL base Prediction done

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


# main view as index
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

        return render_template("index.html", uploaded_file=img_relative_path, bird=predicted_class,
                               confidence=confidence)

    return render_template("index.html")


@app.route('/gallery')
def gallery():
    base_url = "https://ceylonapz.com/birdai/uploads/"
    pic_url = "https://ceylonapz.com/"
    try:
        response = requests.get(base_url)
        soup = BeautifulSoup(response.text, "html.parser")

        image_extensions = [".jpg", ".jpeg", ".png"]
        images = []

        for link in soup.find_all("a"):
            href = link.get("href")
            if href and any(href.lower().endswith(ext) for ext in image_extensions):
                full_url = pic_url + href.lstrip('/')
                images.append(full_url)

        return render_template('gallery.html', images=images)

    except Exception as e:
        return f"Error loading gallery: {str(e)}", 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
