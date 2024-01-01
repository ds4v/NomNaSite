import os
import shutil
import streamlit as st
from urllib.request import urlretrieve
from crnn import CRNN
from dbnet import DBNet


@st.cache_resource(show_spinner='Downloading model weights and vocab.txt...')
def download_assets():
    if not os.path.exists('assets.zip'):
        urlretrieve('https://nomnaocr.000webhostapp.com/assets.zip', 'assets.zip')
    if not os.path.exists('assets'):
        shutil.unpack_archive('assets.zip', 'assets')


@st.cache_resource(show_spinner='Loading model weights...')
def load_models():
    det_model = DBNet()
    rec_model = CRNN()
    det_model.model.load_weights('./assets/DBNet.h5')
    rec_model.model.load_weights('./assets/CRNN.h5')
    return det_model, rec_model


def file_uploader(image_name='test.jpg'):
    uploaded_file = st.file_uploader('Choose a file:', type=['jpg', 'jpeg', 'png'])
    url = st.text_input('Image URL:', 'http://www.nomfoundation.org/data/kieu/1866/page01b.jpg')
    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
        with open(image_name, 'wb') as f:
            f.write(bytes_data)
    elif url: urlretrieve(url, image_name)