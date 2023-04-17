import cv2
import json
import hashlib
import numpy as np
import streamlit as st

from PIL import Image
from streamlit_drawable_canvas import st_canvas
from handler.asset import download_assets, load_models, file_uploader
from handler.bbox import transform_fabric_box, get_patch
from handler.translator import hcmus_translate, hvdic_render


def img2str(cv2_image):
    img_bytes = cv2.imencode('.jpg', cv2_image)[1].tobytes()
    hash_object = hashlib.md5(img_bytes)
    hash_str = hash_object.hexdigest()
    print(hash_str)
    return hash_str
    

st.set_page_config('Digitalize old Vietnamese handwritten script for historical document archiving', '🇻🇳', 'wide')
download_assets()    
det_model, reg_model = load_models()
col1, col2 = st.columns([3, 4])
image_name = 'test.jpg'


with col1:
    file_uploader(image_name)
    mode = st.radio('Box Mode:', ('Drawing', 'Editing'), horizontal=True, label_visibility='collapsed')
    
    with st.spinner('Detecting bounding boxes containing text...'):
        raw_image = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB)
        boxes = det_model.predict_one_page(raw_image)
        
        # https://github.com/andfanilo/streamlit-drawable-canvas/issues/73
        # width, height = raw_image.shape[1], raw_image.shape[0]
        # canvas_width = st_javascript('await fetch(window.location.href).then(response => window.innerWidth)')
        # canvas_height = height * (canvas_width / width) # For responsive canvas
        
        initial_drawing = {'version': '4.4.0', 'objects': [{
            'type': 'rect',
            'left': int(box_pts[0][0]),
            'top': int(box_pts[0][1]),
            'width': int(box_pts[1][0] - box_pts[0][0]),
            'height': int(box_pts[-1][1] - box_pts[0][1]),
            'fill': 'rgba(76, 175, 80, 0.3)',
            'stroke': 'red',
            'strokeWidth': 2,
            'strokeUniform': True,
            # 'cornerSize': 8,
            # 'cornerColor': 'limegreen',
            'transparentCorners': False
        } for box_pts in boxes]}

        canvas_result = st_canvas(
            background_image = Image.open(image_name) if image_name else None,
            fill_color = 'rgba(76, 175, 80, 0.3)',
            width = max(raw_image.shape[1], 1),
            height = max(raw_image.shape[0], 1),
            stroke_width = 2,
            stroke_color = 'red',
            drawing_mode = 'rect' if mode == 'Drawing' else 'transform',
            initial_drawing = initial_drawing,
            update_streamlit = False,
            key = img2str(raw_image)
        )
        
        
with col2:
    json_data = canvas_result.json_data
    canvas_boxes = json_data['objects'] if json_data and 'objects' in json_data else []
    
    with st.spinner('Recognizing text in each bounding box...'):
        for idx, box in enumerate(canvas_boxes[:1]):
            box4pts = transform_fabric_box(box)
            patch = get_patch(raw_image, box4pts)
            text = reg_model.predict_one_patch(patch)
            st.markdown(f'''
                <b>Text {idx + 1:02d}</b>: {text}<br/>
                [hcmus](https://www.clc.hcmus.edu.vn/?page_id=3039): {hcmus_translate(text)}<br/>
                [hvdic](https://hvdic.thivien.net/transcript.php#trans): {hvdic_render(text)}<hr style="margin: 0;"/>
            ''', unsafe_allow_html=True)