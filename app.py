import cv2
import json
import streamlit as st

from PIL import Image
from zipfile import ZipFile
from streamlit_drawable_canvas import st_canvas
from streamlit_javascript import st_javascript

from handler.asset import hash_bytes, download_assets, load_models, retrieve_image
from handler.bbox import generate_initial_drawing, transform_fabric_box, order_boxes4nom, get_patch
from handler.translator import hcmus_translate, hvdic_render
from toolbar import render_toolbar
from style import custom_css


st.set_page_config('Digitalize old Vietnamese handwritten script for historical document archiving', '🇻🇳', 'wide')
st.markdown(custom_css, unsafe_allow_html=True)
download_assets()    
det_model, rec_model = load_models()
col1, col2 = st.columns(2)


with st.sidebar:
    st.image('imgs/cover.jpg')
    st.header('Leverage Deep Learning to digitize old Vietnamese handwritten for historical document archiving')
    st.info("Vietnamese Hán-Nôm digitalization using [VNPF's site](http://www.nomfoundation.org) as collected source")
    
    uploaded_file = st.file_uploader('Choose a file:', type=['jpg', 'jpeg', 'png'])
    url = st.text_input('Image URL:', 'http://www.nomfoundation.org/data/kieu/1866/page01b.jpg')
    image_path = retrieve_image(uploaded_file, url)
    print(image_path)
    
    st.markdown('''
        #### My digitalization series: 
        - [Optical Character Recognition](https://github.com/ds4v/NomNaOCR)
        - [Neural Machine Translation](https://github.com/ds4v/NomNaNMT)
        - [Web Application](https://github.com/ds4v/NomNaSite)
    ''')
    st.markdown('''
        <hr style="margin-top: 0;"/>
        <p align="center">
            <a href="https://github.com/18520339" target="_blank">
                <img src="https://img.shields.io/badge/Quan%20Dang-100000?style=for-the-badge&logo=github" />
           </a>
        </p>
    ''', unsafe_allow_html=True)
    

with col1:
    raw_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    canvas_width = st_javascript('await fetch(window.location.href).then(response => window.innerWidth)')
    # canvas_width = min(canvas_width, raw_image.shape[1])
    canvas_height = raw_image.shape[0] * canvas_width / raw_image.shape[1] # For responsive canvas
    size_ratio = canvas_height / raw_image.shape[0]
    
    with st.spinner('Detecting bounding boxes containing text...'):
        img_bytes = cv2.imencode('.jpg', raw_image)[1].tobytes()
        key = hash_bytes(img_bytes)
        boxes = det_model.predict_one_page(raw_image)
        mode, rec_clicked = render_toolbar(key)
        print(key)

        canvas_result = st_canvas(
            background_image = Image.open(image_path) if image_path else None,
            fill_color = 'rgba(76, 175, 80, 0.3)',
            width = max(canvas_width, 1),
            height = max(canvas_height, 1),
            stroke_width = 2,
            stroke_color = 'red',
            drawing_mode = 'rect' if mode == 'Drawing' else 'transform',
            initial_drawing = generate_initial_drawing(boxes, size_ratio),
            update_streamlit = rec_clicked,
            key = f'canvas_{key}'
        )
        
        
with col2:
    canvas_boxes = []
    if canvas_result.json_data and 'objects' in canvas_result.json_data:
        canvas_boxes = order_boxes4nom([
            transform_fabric_box(obj, size_ratio) 
            for obj in canvas_result.json_data['objects']
        ])

    with st.spinner('Recognizing text in each bounding box...'):
        with ZipFile('data/patches.zip', 'w') as zip_file:
            with open(f'data/data.json', 'w', encoding='utf-8') as json_file:
                saved_json = {
                    'num_boxes': len(canvas_boxes), 
                    'height': raw_image.shape[0], 
                    'width': raw_image.shape[1], 
                    'patches': []
                }
                with open(f'data/data.csv', 'w', encoding='utf-8', newline='') as csv_file:
                    csv_file.write('x1,y1,x2,y2,x3,y3,x4,y4,nom,modern,height,width\n')
                    
                    for idx, box in enumerate(canvas_boxes):
                        patch = get_patch(raw_image, box)
                        nom_text = rec_model.predict_one_patch(patch).strip()
                        modern_text = hcmus_translate(nom_text).strip()
                            
                        with st.expander(f':red[**Text {idx + 1:02d}**:] {nom_text}'):
                            col21, col22 = st.columns([1, 7])
                            with col21:
                                st.image(patch)
                            with col22:
                                points = sum(box.tolist(), [])
                                points = ','.join([str(round(p)) for p in points])
                                saved_json['patches'].append({
                                    'nom': nom_text, 'modern': modern_text, 'points': points, 
                                    'height': str(patch.shape[0]), 'width': str(patch.shape[1])
                                })
                                st.table(saved_json['patches'][-1])
                            
                        st.markdown(f'''
                            [hcmus](https://www.clc.hcmus.edu.vn/?page_id=3039): {modern_text}<br/>
                            [hvdic](https://hvdic.thivien.net/transcript.php#trans): {hvdic_render(nom_text)}
                        ''', unsafe_allow_html=True)

                        encoded_patch = cv2.imencode('.jpg', cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))[1]
                        zip_file.writestr(f'img/{nom_text}.jpg', encoded_patch)
                        csv_file.write(points + f',{nom_text},{modern_text},{patch.shape[0]},{patch.shape[1]}\n')
                json.dump(saved_json, json_file, ensure_ascii=False, indent=4)
                
            zip_file.write('data/data.json')
            zip_file.write('data/data.csv')