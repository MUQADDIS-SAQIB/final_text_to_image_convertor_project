from flask import Flask
from app.routes import main_routes


def create_app():
    app = Flask(__name__)
    app.secret_key = "your_secret_key"  # Needed for flash messages

    # Register blueprints
    app.register_blueprint(main_routes)

    return app
