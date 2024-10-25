import os
import torch
from flask import Flask, render_template, request, send_file, flash, redirect, url_for
from diffusers import StableDiffusionPipeline
from PIL import Image
import re
from io import BytesIO
from huggingface_hub import login
import base64

# Hugging Face Token (Replace with your actual token)
token = "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"  # Update this with the correct token

# Try logging into Hugging Face
try:
    login(token)
    print("Hugging Face login successful.")
except Exception as e:
    print(f"Error logging into Hugging Face: {e}")

# Model ID
model_id = "CompVis/stable-diffusion-v1-4"

# Load the model
pipe = None  # Initialize the pipeline variable for model loading
try:
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.to(device)
    print(f"Model loaded successfully on {device}.")
except Exception as e:
    print(f"Error loading the model: {e}")

# Initialize Flask app
app = Flask(__name__, template_folder='../templates')

# Store generated images
images = []

# Max length for the prompt
MAX_PROMPT_LENGTH = 1000


@app.route('/')
def page():
    return render_template('front_page.html')


@app.route('/generate_image', methods=['POST'])
def generate_image():
    global images
    prompt = request.form['description']

    # Validate the prompt
    if len(prompt) > MAX_PROMPT_LENGTH:
        flash(f'Invalid input: description is too long (max {MAX_PROMPT_LENGTH} characters)', 400)
        return redirect(url_for('page'))

    if not re.match(r"^[A-Za-z0-9 .,!?'-]*$", prompt):
        flash('Invalid input: description contains non-English characters', 400)
        return redirect(url_for('page'))

    # Clear previous images
    images = []

    # Check if the model is loaded
    if pipe is None:
        flash('Error: Model not loaded properly. Please check the logs.', 500)
        return redirect(url_for('page'))


    try:
        # Generate 3 images
        for i in range(3):
            print(f"Generating image {i + 1} with prompt: {prompt}")
            image_result = pipe(prompt, num_inference_steps=25)

            if not image_result or image_result.images is None:
                flash('Error: Failed to generate image', 500)
                return redirect(url_for('page'))

            # Convert image to base64 and store
            image = image_result.images[0]
            buffer = BytesIO()
            image.save(buffer, format='JPEG')
            buffer.seek(0)
            images.append({
                'format': 'jpeg',
                'data': base64.b64encode(buffer.getvalue()).decode('utf-8')
            })

        # Render the result with the images
        return render_template('front_page.html', images=images)

    except Exception as e:
        print(f"Error during image generation: {e}")
        flash(f'Error generating image: {str(e)}', 500)
        return redirect(url_for('page'))



@app.route('/download_image/<int:image_id>', methods=['GET'])
def download_image(image_id):
    global images
    if image_id >= len(images):
        flash('Image ID not found', 404)
        return redirect(url_for('page'))


    try:
        # Get the selected format for download (JPEG or GIF)
        image_format = request.args.get('format', 'jpeg')
        image_data = base64.b64decode(images[image_id]['data'])
        buffer = BytesIO(image_data)

        # Convert to GIF if selected
        if image_format == 'gif':
            image = Image.open(buffer)
            buffer = BytesIO()  # Reset buffer for GIF
            image.save(buffer, format='GIF')
            buffer.seek(0)

        return send_file(buffer, mimetype=f'image/{image_format}', as_attachment=True,
                         download_name=f'image_{image_id + 1}.{image_format}')

    except Exception as e:
        print(f"Error during image download: {e}")
        flash('Error downloading image', 500)
        return redirect(url_for('page'))







# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=5010, use_reloader=False)


