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
import openvino as ov
import openvino.properties as props

app = Flask(__name__)


CLASSES = ["DR", "MH", "ODC", "TSLN", "DN", "MYA", "ARMD"]
DISEASE_THRESHOLD = 0.5

def load_models_keras(model_path):

    model = load_model(model_path)

    return model

def load_models_ov(model_path):

    core = ov.Core()
    core.set_property({props.cache_dir: "./ov_cpu_cache"})
    ov_model = core.read_model(model_path)
    compiled_model = core.compile_model(ov_model)
    
    return compiled_model

def process_image(img, size):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = img / 255.0
    input_tensor = np.expand_dims(img, axis=0)

    return input_tensor


#model_bincl = load_models_keras("retinal_model.keras")
#model_multi = load_models_keras("retinal_model1.keras")

model_bincl = load_models_ov("Models/retinal_model.xml")
model_multi = load_models_ov("Models/retinal_model_multi.xml")


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
    #prediction = model_bincl.predict(processed_image)
    prediction= model_bincl([processed_image])[model_bincl.output(0)]
    confidence_score = float(prediction[0][0])
    result = "Disease" if confidence_score > DISEASE_THRESHOLD else "Healthy"

    diseases = []
    disease_scores = {}
    if result == "Disease":
        processed_image_multi = process_image(image, (224, 224))
        #scores = model_multi.predict(processed_image_multi)[0]
        scores= model_multi([processed_image_multi])[model_multi.output(0)][0]
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
