import os
import base64
from io import BytesIO
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
from huggingface_hub import login

# Constants
MAX_PROMPT_LENGTH = 1000

# Load environment variables
token = os.getenv("HUGGING_FACE_TOKEN")

# Log in to Hugging Face
if token:
    login(token)
else:
    raise ValueError("Hugging Face token is missing.")

# Load Model
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe.to(device)

# Generate images based on prompt
def generate_images(prompt):
    images = []
    for i in range(3):  # Generate 3 images
        image_result = pipe(prompt, num_inference_steps=5)
        if image_result and image_result.images:
            # Convert image to base64
            image = image_result.images[0]
            buffer = BytesIO()
            image.save(buffer, format='JPEG')
            images.append({
                'format': 'jpeg',
                'data': base64.b64encode(buffer.getvalue()).decode('utf-8')
            })
    return images

# Handle image download
def download_image_file(image_id, image_format):
    try:
        image_data = base64.b64decode(images[image_id]['data'])
        buffer = BytesIO(image_data)
        if image_format == 'gif':
            image = Image.open(buffer)
            buffer = BytesIO()
            image.save(buffer, format='GIF')
        buffer.seek(0)
        return buffer
    except Exception as e:
        print(f"Error during image download: {e}")
        return None
