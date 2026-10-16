import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

CLASSES = ['Covid', 'Normal', 'Viral Pneumonia']   # same order as printed in the notebook

model = load_model('best_model.keras')

st.title('COVID-19 Detection from Chest X-ray')
st.write('Upload a chest X-ray image to get a prediction.')

file = st.file_uploader('Upload X-ray', type=['jpg', 'jpeg', 'png'])

if file is not None:
    img = Image.open(file).convert('RGB')
    st.image(img, caption='Uploaded X-ray', width=300)

    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    probs = model.predict(img)[0]

    st.success('Prediction: ' + CLASSES[np.argmax(probs)])
    st.write('Confidence: {:.1f}%'.format(np.max(probs) * 100))
    for name, p in zip(CLASSES, probs):
        st.write(name + ': {:.1f}%'.format(p * 100))
