"""Web app for Retinal Disease Prediction"""

#Import libraries
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
import warnings
warnings.filterwarnings("ignore")

#Disease classes
classes = {1:'DR',2:'MH',3:'ODC',4:'TSLN',5:'DN',6:'MYA',7:'ARMD'}

#Load models
model=tf.keras.models.load_model('retinal_model.keras')
model_multi = tf.keras.models.load_model('retinal_model1.keras')

# Process the image

def processImage(img, size):
    img = cv2.resize(img, size)
    img = img / 255.0  
    img = np.expand_dims(img, axis=0) 

    return img

st.set_page_config(
    page_title="Retinal Disease Detection",
    page_icon = ":eye:",
    initial_sidebar_state = 'auto'
)

hide_streamlit_style = """
	<style>
  #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
  </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.write("""
         # Retina Disease Detection 
         """
         )
file = st.file_uploader("Choose a file: ", type=["png"])
if file is None:
    st.text("Please upload an image file")
else:
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image, use_container_width= True)
    image_size = (128,128)
    processed_image = processImage(image, image_size)
    prediction = model.predict(processed_image)
    score = prediction[0][1]
    result = 'Disease' if score>.50 else 'Healthy'
    
    
    if result == 'Healthy':
        st.write('Healthy retina')

    else:
        st.write('Unhealthy retina')
        st.write('The image shows the retina at risk of the following disease(s)...')
        image_size = (224,224)
        processed_image_multi = processImage(image, image_size)
        prediction_disease = model_multi.predict(processed_image_multi)
        w_index = np.where(prediction_disease == 1)[0]
        disease = [classes.get(prediction_disease[i+1]) for i in w_index]
        if disease == []:
            st.write('Disease not listed')
        else:
            st.write(str(disease))
    
    


