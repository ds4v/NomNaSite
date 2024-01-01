import os
import shutil
import hashlib
import streamlit as st

from urllib.request import urlretrieve
from crnn import CRNN
from dbnet import DBNet


def hash_bytes(bytes_data):
    hash_object = hashlib.sha256(bytes_data)
    hash_str = hash_object.hexdigest()
    return hash_str


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


@st.cache_resource(show_spinner='Retrieving image...')
def retrieve_image(uploaded_file, url):
    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
        image_path = f'./imgs/{hash_bytes(bytes_data)}.jpg'
        with open(image_path, 'wb') as f:
            f.write(bytes_data)
    elif url: 
        bytes_data = url.encode(encoding='utf-8')
        image_path = f'./imgs/{hash_bytes(bytes_data)}.jpg'
        urlretrieve(url, image_path)
    return image_path