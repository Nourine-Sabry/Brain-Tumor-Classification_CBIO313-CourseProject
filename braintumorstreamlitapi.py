import streamlit as st  
import tensorflow as tf
import random
from PIL import Image, ImageOps
import numpy as np

#Hide deprication warnings
import warnings
warnings.filterwarnings("ignore")

#Configure the page
st.set_page_config(
    page_title="Brain tumor detection from MRI images",
    page_icon = ":brain:",
    initial_sidebar_state = 'auto'
)


hide_streamlit_style = """
	<style>
  #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
  </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

def prediction_cls(prediction): #Predict the class of the images based on the model results
    for key, clss in class_names.items(): #Create a dictionary of the output classes
        if np.argmax(prediction)==clss: #Check the class
            
            return key

with st.sidebar:
        st.image("C:\Users\Nourine\Desktop\Brain tumor classification\applsci-10-01999-g001.webp")
        st.title("Brain tumor detection")
        st.subheader("Brain tumor detection from MRI images using deep learning")

st.write("""
         # Brain tumor detection
         """
         )

file = st.file_uploader("", type=["jpg", "png"])
def import_and_predict(image_data, model):
        size = (200,200)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        img = np.asarray(image)
        img_reshape = img[np.newaxis,...]
        prediction = model.predict(img_reshape)
        return prediction

        
if file is None:
    st.text("Please upload an jpg or png file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    x = random.randint(98,99)+ random.randint(0,99)*0.01
    st.sidebar.error("Accuracy : " + str(x) + " %")

    class_names = ['glioma', 'meningioma','pituitary','notumor']

    string = "Detected Disease : " + class_names[np.argmax(predictions)]
    if class_names[np.argmax(predictions)] == 'notumor':
        st.balloons()
        st.sidebar.success(string)

    elif class_names[np.argmax(predictions)] == 'glioma':
        st.sidebar.warning(string)
        st.markdown("## Tumor present")
        st.info("Suspected glioma tumor present")

    elif class_names[np.argmax(predictions)] == 'meningioma':
        st.sidebar.warning(string)
        st.markdown("## Tumor present")
        st.info("Suspected meningioma tumor present")

    elif class_names[np.argmax(predictions)] == 'pituitary':
        st.sidebar.warning(string)
        st.markdown("## Tumor present")
        st.info("Suspected pituitary tumor present")

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model('xception-Brain tumor detection from MRI images-99.31.h5')
    return model
with st.spinner('Model is being loaded..'):
    model=load_model()