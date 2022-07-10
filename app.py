import os
import cv2
import shutil
import numpy as np
import pandas as pd
import streamlit as st
from urllib.request import urlretrieve
from utils import load_models, get_patch


@st.cache
def download_assets():
    if os.path.exists('assets.zip'): return
    urlretrieve('https://nomnaftp.000webhostapp.com/assets.zip', 'assets.zip')
    shutil.unpack_archive('assets.zip', 'assets')
    
    
st.set_page_config(page_title='NomNaOCR Demo', page_icon="🇻🇳", layout='wide')
uploaded_file = st.file_uploader("Choose a file")
url = st.text_input('Image Url:', 'http://www.nomfoundation.org/data/kieu/1866/page01a.jpg')

st.write('')
download_assets()    
det_model, reg_model = load_models()
col1, col2, col3 = st.columns(3)
    
with col1:
    st.header('Input Image:')
    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
        st.image(bytes_data)
        with open('test.jpg', 'wb') as f:
            f.write(bytes_data)
    elif url: 
        urlretrieve(url, 'test.jpg')
        st.image('test.jpg')

with col2:
    st.header('Text Detection:')
    with st.spinner('Detect bounding boxes contain text'):
        raw_image, boxes, _ = det_model.predict_one_page('test.jpg')
        boxes = sorted(boxes, key=lambda box: (box[:, 0].max(), box[:, 1].min()))
        image = raw_image.copy()
        
        for box in boxes:
            box = box.astype(np.int32)[np.newaxis]
            cv2.polylines(image, box, color=(0, 255, 0), thickness=2, isClosed=True)
    st.image(image)
    
with col3:
    st.header('Text Recognition:')
    with st.spinner('Recognize text in each predicted bounding box'):
        for idx, box in enumerate(boxes):
            patch = get_patch(raw_image, box)
            text = reg_model.predict_one_patch(patch)
            st.caption(f'{idx + 1}. {text}')        