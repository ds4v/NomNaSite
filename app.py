import cv2
import numpy as np
import pandas as pd
import urllib.request
import streamlit as st

from crnn import CRNN
from dbnet import DBNet
from utils import get_patch


det_model = DBNet()
reg_model = CRNN()
det_model.model.load_weights('./assets/DBNet.h5')
reg_model.model.load_weights('./assets/CRNN.h5')

st.set_page_config(layout="wide")
uploaded_file = st.file_uploader("Choose a file")
url = st.text_input('Image Url:', 'http://www.nomfoundation.org/data/kieu/1866/page01a.jpg')
st.write('')
col1, col2, col3 = st.columns(3)
    
with col1:
    st.header('Input Image:')
    if url: urllib.request.urlretrieve(url, './test.jpg')
    elif uploaded_file is not None:
        bytes_data = uploaded_file.read()
        with open('./test.jpg', 'wb') as f:
            f.write(bytes_data)
    st.image('./test.jpg')

raw_image, boxes, _ = det_model.predict_one_page('./test.jpg')
image = raw_image.copy()
boxes = sorted(boxes, key=lambda box: (box[:, 0].max(), box[:, 1].min()))
texts = []

for box in boxes:
    patch = get_patch(raw_image, box)
    box = box.astype(np.int32)[np.newaxis]
    cv2.polylines(image, box, color=(0, 255, 0), thickness=2, isClosed=True)
    texts.append(reg_model.predict_one_patch(patch))

with col2:
    st.header('Text Detection:')
    st.image(image)
    
with col3:
    st.header('Text Recognition:')
    for idx, text in enumerate(texts): 
        st.caption(f'{idx + 1}. {text}')