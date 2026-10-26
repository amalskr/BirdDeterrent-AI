# BirdDeterrent-AI

Inference server and deployed model checkpoint for a proof-of-concept (PoC) AI and IoT bird deterrent for paddy fields in Sri Lanka. The model classifies a camera frame as `target` (pest birds: Indian peafowl, *Pavo cristatus*, and white-rumped munia, *Lonchura striata*) or `none` (other bird species).

This repository is the artefact referenced in the paper:

> A. Ranasinghe and J. Wickramarathne, "AI and IoT-Based Smart Deterrent System for Protecting Paddy Fields from Birds: A Proof-of-Concept Field Deployment Evaluation," in *Proc. 8th IEEE International Conference on Advancements in Computing (ICAC 2026)*, Colombo, Sri Lanka, Dec. 2026 (to appear).

The code is kept exactly as it ran during the field deployment (March to April 2025), including a known preprocessing defect described below. It is a research artefact, not a production system.

## System overview

```
Ultrasonic sensor (ESP32) ──> servo sweep (always, before classification)
        │
        └─ MQTT ──> ESP32-CAM capture ──HTTPS──> web host (PHP upload)
                                                     │
                                          HTTPS (image URL)
                                                     ▼
                                   This Flask server: MobileNetV2 classifier
                                                     │
                                       label ──MQTT──> ESP32: 5 servo sweeps + LED
```

The ESP32 firmware is not part of this repository.

## Repository contents

- `app.py`: Flask inference server
- `model/wpb_7.keras`: deployed checkpoint (epoch 7, about 19 MB)
- `templates/`: HTML pages for manual upload testing and the image gallery
- `requirements.txt`: Python dependencies (TensorFlow 2.18, Keras 3.9, Flask 3.1, OpenCV)

## Model

- Architecture: MobileNetV2 (ImageNet weights), last 20 layers trainable, global average pooling, batch normalisation, dropout 0.5, one sigmoid unit
- Input: 224 x 224 RGB, scaled to [0, 1]
- Output: probability of `target`; class mapping `{0: "none", 1: "target"}`
- Training data: [25 Indian Bird Species with 22.6k Images](https://www.kaggle.com/datasets/arjunbasandrai/25-indian-bird-species-with-226k-images) (Kaggle, Version 4), regrouped into two classes, plus a small number of web-sourced munia flock images in `target`

| Partition | none | target | Total |
|---|---|---|---|
| Training | 17,361 | 747 | 18,108 |
| Validation | 4,340 | 186 | 4,526 |

## Evaluation

Measured on the 4,526 validation images, without augmentation, at threshold 0.5:

| Preprocessing | Precision | Recall | F1 | FPR | Average precision |
|---|---|---|---|---|---|
| RGB (as in training) | 0.767 | 0.973 | 0.858 | 0.013 | 0.971 |
| BGR (as deployed in `app.py`) | 0.460 | 0.855 | 0.598 | 0.043 | 0.736 |

Metrics are for the `target` class. The validation set was also used to select the checkpoint, and the split is per image, so these figures are optimistic. No labelled set of ESP32-CAM field frames exists.

## Known issue: colour channel order

`app.py` loads images with `cv2.imread()`, which returns BGR, and passes them to the model without conversion. The model was trained on RGB images. This defect is kept on purpose because it is part of what the paper reports. If you reuse the server, convert the channels before inference:

```python
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224)) / 255.0
```

The decision threshold of 0.5 was not tuned. On RGB input, a threshold of 0.9 halves false positives (55 to 30) for one additional missed `target` image.

## Running locally

```bash
git clone https://github.com/amalskr/BirdDeterrent-AI.git
cd BirdDeterrent-AI
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
mkdir -p static/uploads
python app.py
```

The server starts on `http://localhost:5000`.

## API

| Method | Endpoint | Description |
|---|---|---|
| GET, POST | `/` | Web page to upload an image and view the prediction |
| GET | `/predict?img_url=<url>` | Downloads the image at `img_url` and classifies it |
| GET | `/gallery` | Lists images from the deployment upload directory |
| GET | `/test?name=<value>` | Health check that echoes `name` |

Example:

```bash
curl "http://localhost:5000/predict?img_url=https://example.com/bird.jpg"
```

```json
{
  "status": "success",
  "attack": "target",
  "confidence": "97.42%"
}
```

`confidence` is the model's probability of `target`, so a `none` result shows a low value.

## Security notes

This server was written for a supervised PoC and should not be exposed publicly as is:

- `app.run(..., debug=True)` enables the Werkzeug debugger; set `debug=False` or use a production WSGI server such as Gunicorn
- `/predict` fetches any URL it is given, with no allow list or authentication
- `/gallery` reads a public directory listing on the deployment web host
- uploaded images are kept with no retention limit

## Limitations

- One node, one paddy plot, one two-week sowing window
- Binary output; the model cannot distinguish peafowl from munia, so it cannot select a species-specific deterrent
- The `none` class contains only other bird species, with no bird-free scenes
- Field accuracy is unknown because no labelled field frames were collected

## Citation

```bibtex
@inproceedings{ranasinghe2026birddeterrent,
  author    = {Ranasinghe, Amal and Wickramarathne, Jagath},
  title     = {{AI} and {IoT}-Based Smart Deterrent System for Protecting Paddy Fields from Birds: A Proof-of-Concept Field Deployment Evaluation},
  booktitle = {Proc. 8th IEEE International Conference on Advancements in Computing (ICAC)},
  address   = {Colombo, Sri Lanka},
  year      = {2026},
  note      = {to appear}
}
```

## Authors

- Amal Ranasinghe, Sri Lanka Institute of Information Technology ([ORCID 0009-0002-3089-6445](https://orcid.org/0009-0002-3089-6445))
- Jagath Wickramarathne, Department of Computer Science and Software Engineering, Sri Lanka Institute of Information Technology ([ORCID 0000-0002-1681-1495](https://orcid.org/0000-0002-1681-1495))
