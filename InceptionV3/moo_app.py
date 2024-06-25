import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

def import_and_predict(image_data, model):
    size = (224, 224)       # Set image size
    image = ImageOps.fit(image_data, size, method=Image.LANCZOS)        # Prepare image with anti-aliasing
    image = image.convert('RGB')        # Convert image to RGB
    image = np.asarray(image)           # Convert image into array
    image = (image.astype(np.float32 / 255.0))      # Create image array matrix
    img_reshape = image[np.newaxis, ...]            # np.newaxis will create new dimension
    prediction = model.predict(img_reshape)         # Give predicted output based on the input image
    return prediction                               # Return prediction

# Load the trained model
model = tf.keras.models.load_model('C:\\Users\\dylan\\Documents\\AY2024-25\\MLAI\\Mini Project\\MangoOrOrange\\model.py')
pre_trained_model = InceptionV3(input_shape = (75, 75, 3),
                                include_top = False,
                                weights = 'imagenet')

# Create Streamlit UI Elements
st.write("""# Mango or Orange Classifier""")
st.write("This is a image classification web app to predict Mango, Orange or None")

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
#

if file is None:
    st.text("No image file was selected!")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("It is a mango!")
    elif np.argmax(prediction) == 1:
        st.write("It is an orange!")
    else:
        st.write("Unknown :(")
        
    st.text("Probability (0: Mango, 1: Orange, 2: Unknown)")
    st.write(prediction)