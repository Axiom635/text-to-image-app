import gradio as gr
import requests
import os
from PIL import Image
from io import BytesIO

# Load Hugging Face token from environment variable
HF_TOKEN = os.getenv("HF_TOKEN")

# Hugging Face Inference API endpoint (Stable Diffusion 2.1 is CPU-friendly)
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def generate_image(prompt):
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})

    if response.status_code == 200 and "image" in response.headers.get("content-type", ""):
        return Image.open(BytesIO(response.content))
    else:
        return f"Error {response.status_code}: {response.text[:300]}"

demo = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(label="Enter your prompt"),
    outputs=gr.Image(type="pil"),
    title="Text-to-Image Generator",
    description="Generate images from text using Stable Diffusion via Hugging Face Inference API"
)

if __name__ == "__main__":
    # Required for Render: bind to external port
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))





