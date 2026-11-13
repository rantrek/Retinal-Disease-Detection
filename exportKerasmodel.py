import openvino as ov
from keras.models import load_model

model = load_model("retinal_model1.keras")

model.export("temp_saved_model", format="tf_saved_model")

ov_model = ov.convert_model("temp_saved_model")
ov.save_model(ov_model, "./Models/retinal_model_multi.xml")

