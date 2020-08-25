"""Streamlit web app"""

import streamlit as st
import joblib, os
import numpy as np
from PIL import Image
import requests
import urllib
import datetime,time

WIDTH = 300
HEIGHT = 300
IMG_PATH = "imgs"
URI= 'https://raw.githubusercontent.com/Chloejay/image_caption_app/master/app/'

st.set_option("deprecation.showfileUploaderEncoding", False)
st.beta_set_page_config( page_title="Image translation app",
                        page_icon="",
                        layout="wide",
                        initial_sidebar_state="expanded",
                        )
    
@st.cache(show_spinner = False)
def get_file_content_as_string(app_file_path):
    url = URI + app_file_path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

def run_the_app():
    st.markdown("## `#TODO`")
    
def render_default():
    st.title("Translate Image to Text")
    st.markdown("--"*50)
    st.text("Display image metrics etc")

def get_time():
    st.sidebar.date_input("Today is", datetime.datetime.now())

def upload_img():
    uploaded_file = st.file_uploader("Please choose an image to upload...", type= ["jpeg", "jpg"], multiple_files= True)
    if uploaded_file:
        img = np.array(Image.open(uploaded_file))
        st.image(img, width = WIDTH, height = HEIGHT, caption = "Image", use_column_width = True) 
        st.success("###### success load image")
        st.write("")
        st.write("Generate text from image features...")
        # TODO

def load_readme(img_path):
    """
    add some model paper results here for explantion.
    """
    st.title("TODO can write model theory...")
    # pbar= st.progress(0)
    all_imgs = list()
    for i in [x for x in os.listdir(img_path) if x.startswith("img")]:
        img = Image.open(os.path.join(img_path, i))
        all_imgs.append(img)
    st.image(all_imgs, width = WIDTH)
    st.success("###### success render example image")
    # pbar.progress(p+1)
    
def load_tensorboard():
    st.markdown("## below is the model result from the tensorboard")
    with st.echo():
	    import pandas as pd 
	    df = pd.DataFrame()

# app 
def main():
    app_mode = st.sidebar.selectbox("Choose the app mode",["APP model infos", 
                                                           "Run the app", 
                                                           "Show the source code"])

    if app_mode == "APP model infos":
        st.sidebar.info("This is a app about ..., at this moment this model will only to be tried one model from paper\
            Show and Tell: A Neural Image Caption Generator, which applies transfer learning, \
            it layers up CNN and RNN models to translate image features to text. Model training is based on COCO datasets(13G) on AWS EC2.")
        render_default()
        load_readme(IMG_PATH)

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
    
    # placeholder st.empty() to switch with text or image.
    
    # to have multiply models to be selected
    # TODO get one more model
    models = ["cnn_lstm", "nlp"]
    model_choice = st.selectbox("Select Model", models)
    st.balloons()
    
    @st.cache
    def load_model(model_file):
        loaded_model = joblib.load(open(os.path.join(model_file), "rb"))
        return loaded_model

    # TODO
    if model_choice =="cnn_lstm":
        # load model
        predictor = load_model("models/...")
        prediction = predictor.predict(vect_text)
        st.write(prediction)
        
    if model_choice=="nlp":
        predictor = load_model("models/...")
        prediction = predictor.predict(vect_text)
        st.write(prediction)
    else:
        st.info("please choose one model then continue...")
        

    if st.button("Translate"):
        run_model()
    else:
        run_inference_result()
    
    
if __name__ == "__main__":
    main()