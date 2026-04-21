import streamlit as st
from model_helper import predict


st.title("Vehicle Damage Detection")
uploaded_file=st.file_uploader("Upload the file",type=["jpg","png"]) #file uploader ie taking image as an input
if uploaded_file:
    image_path="temp_file.jpg"
    with open(image_path,"wb") as f: # storing the binary data of an image in another temporary file ,w means wirte b means binary
        f.write(uploaded_file.getbuffer()) #gives the binary bytes of the uploaded  image
        st.image(uploaded_file,caption="uploaded File",width="content") #displayin the file
        prediction=predict(image_path)
        st.info(f"Predicted Class:{prediction}")
