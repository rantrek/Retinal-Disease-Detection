"""Flask inference API (Backend) for Retinal Disease Detection Web App"""

"""Streamlit UI calls the Flask inference API.

Start the API first:
    python Retinal_Disease_Flask_API.py

Then run:
    streamlit run Retinal_Disease_Streamlit_Client.py"""

from flask import Flask, request, jsonify
import cv2
import numpy as np
from keras.models import load_model

app = Flask(__name__)


CLASSES = ["DR", "MH", "ODC", "TSLN", "DN", "MYA", "ARMD"]
DISEASE_THRESHOLD = 0.5

model = load_model("retinal_model.keras")
model_multi = load_model("retinal_model1.keras")


def process_image(img, size):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = img / 255.0
    return np.expand_dims(img, axis=0)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    filename = (file.filename or "").lower()
    content_type = (file.content_type or "").lower()

    if not (filename.endswith(".png") or content_type in ("image/png", "application/octet-stream")):
        return jsonify({"error": "Invalid file type. Please upload a .png image."}), 400

    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Invalid image file"}), 400

    processed_image = process_image(image, (128, 128))
    confidence_score = float(model.predict(processed_image)[0][0])
    result = "Disease" if confidence_score > DISEASE_THRESHOLD else "Healthy"

    diseases = []
    disease_scores = {}
    if result == "Disease":
        processed_image_multi = process_image(image, (224, 224))
        scores = model_multi.predict(processed_image_multi)[0]
        disease_scores = {name: float(score) for name, score in zip(CLASSES, scores)}
        diseases = [name for name, score in disease_scores.items() if score >= DISEASE_THRESHOLD]

    return jsonify({
        "prediction": result,
        "confidence_score": confidence_score,
        "diseases": diseases,
        "disease_scores": disease_scores,
    })


if __name__ == "__main__":
    app.run()
