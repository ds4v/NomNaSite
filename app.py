import cv2
import numpy as np
import streamlit as st
from urllib.request import urlretrieve
from utils import download_assets, load_models, get_patch, get_phonetics

    
st.set_page_config('Digitalize old Vietnamese handwritten script for historical document archiving', '🇻🇳', 'wide')
col1, col2 = st.columns([5, 4])

with col1:
    st.video('https://user-images.githubusercontent.com/50880271/178230816-c39b5cc7-38e9-4bf3-9803-8e12f286b9fd.mp4')
    
with col2:
    uploaded_file = st.file_uploader('Choose a file', type=['jpg', 'jpeg', 'png'])
    url = st.text_input('Image URL:', 'http://www.nomfoundation.org/data/kieu/1866/page01b.jpg')
    st.markdown('''
        ### Digitalize old Vietnamese handwritten script for historical document archiving
        Vietnamese Hán-Nôm digitalization using [VNPF's site](http://www.nomfoundation.org) as collected source
    ''', unsafe_allow_html=True)

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
    with st.spinner('Detecting bounding boxes contain text...'):
        raw_image, boxes, _ = det_model.predict_one_page('test.jpg')
        boxes = sorted(boxes, key=lambda box: (box[:, 0].max(), box[:, 1].min()))
        image = raw_image.copy()

        for idx, box in enumerate(boxes):
            box = box.astype(np.int32)
            org = (box[3][0] + box[0][0])//2, (box[3][1] + box[0][1])//2
            
            cv2.polylines(image, [box], color=(255, 0, 0), thickness=1, isClosed=True)
            cv2.putText(
                image, str(idx), org, cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.8, color=(0, 0, 255), thickness=2
            )
        st.image(image)
    
with col3:
    st.header('Text Recognition:')
    table = st.table({'Texts': [], 'Phonetics': []})
    
    with st.spinner('Recognizing text in each predicted bounding box...'):
        for idx, box in enumerate(boxes):
            patch = get_patch(raw_image, box)
            text = reg_model.predict_one_patch(patch)
            phonetics = ' '.join([
                d['o'][0] if d['t'] == 3 and len(d['o']) > 0 else '[UNK]' 
                for d in get_phonetics(text)
            ]).strip()
            table.add_rows({'Texts': [text], 'Phonetics': [phonetics[0].upper() + phonetics[1:]]})