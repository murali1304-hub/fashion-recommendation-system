import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import cv2

feature_list = np.array(pickle.load(open('featurevector.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('Smart Fashion Recommendation')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def extract_feature(img_path, model):
    img=cv2.imread(img_path)
    img=cv2.resize(img, (224,224))
    img=np.array(img)
    expand_img=np.expand_dims(img, axis=0)
    pre_img=preprocess_input(expand_img)
    result=model.predict(pre_img).flatten()
    normalized=result/norm(result)
    return normalized

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

# steps
# file upload -> save
st.markdown("<h1 style='text-align: center; font-size: 2em; font-weight: bold;'>Recommendation</h1>", unsafe_allow_html=True)
st.write("")
uploaded_file = st.file_uploader("Upload an image")
print(uploaded_file)
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display the file
        display_image = Image.open(uploaded_file)
        resized_img = display_image.resize((200, 200))
        st.image(resized_img)
        # feature extract
        features = extract_feature(os.path.join("uploads",uploaded_file.name),model)
        #st.text(features)
        # recommendention
        indices = recommend(features,feature_list)
        # show
        col1,col2,col3,col4,col5 = st.columns(5)

        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])
    else:
        st.header("Some error occured while uploading the file")
st.write("")
st.markdown("<h1 style='text-align: center; font-size: 2em; font-weight: bold;'>Wardrobe</h1>", unsafe_allow_html=True)
st.write("")
if st.button("View wardrobe"):
    uploaded_images = os.listdir("uploads")
    
    if not uploaded_images:
        st.write("No images found in the 'uploads' folder.")
    else:
        # Open a new page
        st.markdown("<h1 style='text-align: center;'>Uploaded Images</h1>", unsafe_allow_html=True)
        
        # Display images on the new page
        for image_file in uploaded_images:
            image_path = os.path.join("uploads", image_file)
            st.image(image_path, caption=image_file, use_column_width=True)

web_chat_integration_code="""
      <script>
  window.watsonAssistantChatOptions = {
    integrationID: "b33ea006-a54f-493f-bfe5-87c8fed75c36", // The ID of this integration.
    region: "eu-gb", // The region your integration is hosted in.
    serviceInstanceID: "729be6b9-d00c-4f7a-a899-23218336f6a1", // The ID of your service instance.
    onLoad: async (instance) => { await instance.render(); }
  };
  setTimeout(function(){
    const t=document.createElement('script');
    t.src="https://web-chat.global.assistant.watson.appdomain.cloud/versions/" + (window.watsonAssistantChatOptions.clientVersion || 'latest') + "/WatsonAssistantChatEntry.js";
    document.head.appendChild(t);
  });
</script>
"""

st.components.v1.html(web_chat_integration_code, height=600, width=800)

