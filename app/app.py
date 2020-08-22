"""Streamlit web app"""

import streamlit as st
import joblib, os
import numpy as np
from PIL import Image
import requests
import urllib
import datetime,time
# from retinaface.pre_trained_models import get_model
# from retinaface.utils import vis_annotations

st.set_option("deprecation.showfileUploaderEncoding", False)
    

@st.cache(show_spinner=False)
def get_file_content_as_string(app_file_path):
    url = 'https://github.com/Chloejay/detection_api/tree/master/app' + app_file_path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

def run_the_app():
    st.markdown("## `#TODO`")
    
def render_default():
    st.title("Translate Image to Text")
    st.markdown("--"*50)
    st.text("Display image metrics etc")

def get_time():
    st.sidebar.date_input("Today is",datetime.datetime.now())

def upload_img():
    uploaded_file = st.file_uploader("Please choose an image to upload...", type=["jpeg", "jpg"], multiple_files=True)
    if uploaded_file:
        img = np.array(Image.open(uploaded_file))
        st.image(img, width = 300, height = 300, caption="Image") #use_column_width=True
        # st.image(img, width = 400, height= 400, caption= "Image2")
        st.success("###### success load image")
        st.write("")
        st.write("Generate text from image features...")
        # TODO

def load_readme():
    """
    add some model paper results here for explanation.
    """
    st.title("let's write some model theory and one image to display 'translation'!")
    
def load_tensorboard():
    st.markdown("## below is the model result from the tensorboard")
    with st.echo():
	    import pandas as pd 
	    df = pd.DataFrame()
    

# app 
app_mode = st.sidebar.selectbox("Choose the app mode",["APP model infos", "Run the app", "Show the source code"])

if app_mode == "APP model infos":
    st.sidebar.success("This is a app about ..., at this moment this model will only take one model from paper,\
                       which just shows the model result by using transfer learning, it layers up CNN and \
                           RNN models to translate image features to text.")
    render_default()
    load_readme()
    
elif app_mode == "Show the source code":
    st.sidebar.code(get_file_content_as_string("app.py"))
    render_default()
    load_tensorboard()
    # load tensorboard dir log for model results
    
elif app_mode == "Run the app":
    get_time()
    render_default()
    run_the_app()
    upload_img()
    

# have multiply models to be selected
# TODO get one more model
models = ["cnn_lstm"]
model_choice = st.selectbox("Select Model", models)

@st.cache
def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), "rb"))
    return loaded_model

if model_choice =="cnn_lstm":
    predictor = load_model("models/...")
    prediction = predictor.predict(vect_text)
    st.write(prediction)

