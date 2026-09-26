# Retinal-Disease-Detection

## About the Dataset

The dataset can be downloaded from https://www.kaggle.com/datasets/andrewmvd/retinal-disease-classification

## Objective

The purpose of this project:
1. Build a model that classifies between healthy and unhealthy retinas. (Binary classifier)
2. Build a model that further identifies the disease(s) amongst the images with unhealthy retinas. (Multi-label classifier)
3. Develop a web application that first predicts whether the retina is healthy or unhealthy and then if unhealthy, further identifies the disease(s), given a single image. 

## Program

All code was written in Python and comprise four files. 
1. RetinalDiseaseClassification.ipynb - This Jupyter notebook trained the binary classifier and saved the model.
2. RetinalDiseaseMultilabel.ipynb - This notebook trained the multi-label classifier and saved the model.
3. Retinal_Disease_Prediction_API.py - This python file loads the models and runs inferences, utilizing the Flask API. 
4. Retinal_Disease_Prediction_UI.py - This python file runs the streamlit UI that loads the image, calls the inference API and displays the results. 
5. Retinal_Disease_Prediction_App.py - This python file runs the entire application in streamlit, both acting as UI and running inferences using the models. This is an older file and is slower to load compared to the Streamlit UI + Flask API files. 

Models were trained in the Google Colab environment. To get both models, run both Jupyter notebooks.

To run the application,first start the Flask API file first and then the streamlit UI in your local system. 

## Techniques

   - Data processing
   - Data Visualization
   - Machine Learning
   - Deep Learning 
   - Web app development (Backend (API), Frontend (UI))
      
## Algorithms 

   - Convolution Neural Network (CNN)  

## Libraries
  
   - Keras
   - TensorFlow
   - Pandas
   - Matplotlib
   - NumPy
   - Scikit-Learn
   - Seaborn
   - OpenCV
   - Flask
   - Streamlit

