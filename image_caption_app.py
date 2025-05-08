githubimport streamlit as st
from PIL import Image
import requests
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Load BLIP model and processor
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()

# Set Streamlit page config
st.set_page_config(page_title="Image Caption Generator", layout="centered")
st.title("🖼️ Image Caption Generator using BLIP")

# Input URL from user
image_url = st.text_input("Paste an image URL below:")

if image_url:
    try:
        # Load image from URL
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        st.image(image, caption="Input Image", use_column_width=True)

        # Generate caption
        with st.spinner("Generating caption..."):
            inputs = processor(image, return_tensors="pt")
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)

        st.success("Caption Generated!")
        st.markdown(f"### 📝 Caption: *{caption}*")

    except Exception as e:
        st.error(f"Error loading or processing image: {e}")
