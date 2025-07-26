import gradio as gr
import requests
import os
from PIL import Image
from io import BytesIO

# Get Hugging Face API token from environment variable
HF_TOKEN = os.getenv("HF_TOKEN")

# Hugging Face Stable Diffusion API
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# Function to generate image from prompt
def generate_image(prompt):
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        return f"Error {response.status_code}: {response.text}"

# Gradio UI
demo = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(label="Enter your prompt"),
    outputs=gr.Image(type="pil"),
    title="Text-to-Image Generator",
    description="Generate images from text using Stable Diffusion via Hugging Face Inference API"
)

# Launch app
if __name__ == "__main__":
    demo.launch()
