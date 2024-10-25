import os
from flask import Blueprint, render_template, request, redirect, url_for, flash, send_file
from app.utils import generate_images, download_image_file, MAX_PROMPT_LENGTH

main_routes = Blueprint('main_routes', __name__)

@main_routes.route('/')
def page():
    return render_template('front_page.html')

@main_routes.route('/generate_image', methods=['POST'])
def generate_image():
    prompt = request.form['description']
    # Validate the prompt
    if len(prompt) > MAX_PROMPT_LENGTH:
        flash(f'Invalid input: description is too long (max {MAX_PROMPT_LENGTH} characters)', 'error')
        return redirect(url_for('main_routes.page'))

    if not prompt.isascii():
        flash('Invalid input: description contains non-English characters', 'error')
        return redirect(url_for('main_routes.page'))

    # Generate images and display on the front page
    images = generate_images(prompt)
    return render_template('front_page.html', images=images)

@main_routes.route('/download_image/<int:image_id>', methods=['GET'])
def download_image(image_id):
    image_format = request.args.get('format', 'jpeg')
    buffer = download_image_file(image_id, image_format)
    if buffer:
        return send_file(buffer, mimetype=f'image/{image_format}', as_attachment=True, download_name=f'image_{image_id}.{image_format}')
    else:
        flash('Error downloading image', 'error')
        return redirect(url_for('main_routes.page'))
