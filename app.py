import os
import cv2
import numpy as np
from flask import Flask, render_template, request
from tensorflow import keras
from datetime import datetime
from multiprocessing import Value

# declare counter variable
counter = Value('i', 0)

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


def save_img(img):
    with counter.get_lock():
        counter.value += 1
        count = counter.value
    img_dir = "esp32_imgs"
    if not os.path.isdir(img_dir):
        os.mkdir(img_dir)
    cv2.imwrite(os.path.join(img_dir,"img_"+str(count)+".jpg"), img)
# print("Image Saved", end="\n") # debug

@app.route('/upload', methods=['POST','GET'])
def upload():
    received = request
    img = None
    if received.files:
        print(received.files['imageFile'])
        # convert string of image data to uint8
        file  = received.files['imageFile']
        nparr = np.fromstring(file.read(), np.uint8)
        # decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        save_img(img)

        return "[SUCCESS] Image Received", 201
    else:
        return "[FAILED] Image Not Received", 204

@app.route('/uploadOld', methods=['POST'])
def upload_image():
    if 'imageFile' not in request.files:
        return '❌ No imageFile field in request', 400

    image = request.files['imageFile']

    if image.filename == '':
        return '❌ No selected file', 400

    # Save image with timestamp
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    filename = datetime.now().strftime("esp32_%Y%m%d_%H%M%S.jpeg")
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    image.save(filepath)

    return '✅ Image uploaded successfully!', 200

@app.route('/gallery')
def gallery():
    files = os.listdir(UPLOAD_FOLDER)
    images = [f"/static/uploads/{file}" for file in files if file.lower().endswith(('.jpg', '.jpeg', '.png'))]
    return render_template('gallery.html', images=images)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)