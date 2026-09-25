"""Web app for Retinal Disease Prediction"""

import os
import streamlit as st
import cv2
import numpy as np
import requests


API_URL = os.environ.get("RETINAL_API_URL") 

st.set_page_config(
    page_title="Retinal Disease Detection",
    page_icon=":eye:",
    initial_sidebar_state="auto",
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
#st.caption(f"Predictions are served by Flask at `{API_URL}`")

file = st.file_uploader("Choose a file: ", type=["png"])
if file is None:
    st.text("Please upload an image file")
else:
    image_bytes = file.getvalue()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        st.error("Could not read that image. Please upload a valid .png file.")
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image, use_container_width=True)

        try:
            response = requests.post(
                f"{API_URL}/predict",
                files={"file": (file.name, image_bytes, "image/png")},
                timeout=60,
            )
            data = response.json()
        except requests.exceptions.ConnectionError:
            st.error(
                "Could not reach the Flask API. Start it with "
                "`python Retinal_Disease_Flask_API.py` and try again."
            )
        except requests.exceptions.RequestException as exc:
            st.error(f"Request failed: {exc}")
        else:
            if response.status_code != 200:
                st.error(data.get("error", "Prediction failed."))
            elif data.get("prediction") == "Healthy":
                st.write("Healthy retina")
                st.write(f"Confidence score: {data['confidence_score']:.4f}")
            else:
                st.write("Unhealthy retina")
                st.write(f"Confidence score: {data['confidence_score']:.4f}")
                st.write("The image shows the retina at risk of the following disease(s)...")
                diseases = data.get("diseases") or []
                if not diseases:
                    st.write("Disease not listed")
                else:
                    st.write(", ".join(diseases))
